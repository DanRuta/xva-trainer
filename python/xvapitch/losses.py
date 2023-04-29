import torch
import torch.nn as nn
import torch.nn.functional as F


# from TTS.utils.audio import TorchSTFT
# from TTS.tts.utils.helpers import sequence_mask

try:
    from python.xvapitch.audio import TorchSTFT
    from python.xvapitch.util import sequence_mask
    from python.xvapitch.model import ReversalClassifier
except:
    from audio import TorchSTFT
    from util import sequence_mask
    from model import ReversalClassifier

class VitsGeneratorLoss(nn.Module):
    # def __init__(self, c: Coqpit):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kl_loss_alpha = 1.0# c.kl_loss_alpha
        self.gen_loss_alpha = 1.0#c.gen_loss_alpha
        self.feat_loss_alpha = 1.0#c.feat_loss_alpha
        self.dur_loss_alpha = 1.0#c.dur_loss_alpha
        self.mel_loss_alpha = 45#c.mel_loss_alpha
        self.spk_encoder_loss_alpha = 1.0#c.speaker_encoder_loss_alpha
        self.stft = TorchSTFT(
            # c.audio.fft_size,
            1024,
            # c.audio.hop_length,
            256,
            # c.audio.win_length,
            1024,
            # sample_rate=c.audio.sample_rate,
            sample_rate=22050,
            # mel_fmin=c.audio.mel_fmin,
            mel_fmin=0,
            # mel_fmax=c.audio.mel_fmax,
            mel_fmax=8000,
            # n_mels=c.audio.num_mels,
            n_mels=80,
            use_mel=True,
            do_amp_to_db=True,
        )

        # self.output_log_path = c.output_log_path

        self.TEMP_VIZ_INTERVAL_COUNTER = 0


        # self.pitch_predictor_loss_scale = 0.05
        # self.energy_predictor_loss_scale = 0.05
        self.pitch_predictor_loss_scale = 0.1#*2 # The global loss is half that from FastPitch
        self.energy_predictor_loss_scale = 0.1

        # self.mel_loss_alpha = self.mel_loss_alpha * 2

        # self.energy_predictor_loss_scale = self.args.energy_predictor_loss_scale



    @staticmethod
    def feature_loss(feats_real, feats_generated):
        loss = 0
        for dr, dg in zip(feats_real, feats_generated):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    @staticmethod
    def generator_loss(scores_fake):
        loss = 0
        gen_losses = []
        for dg in scores_fake:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    @staticmethod
    def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
        """
        z_p, logs_q: [b, h, t_t]
        m_p, logs_p: [b, h, t_t]
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl_sample_wise = kl * z_mask
        kl = torch.sum(kl_sample_wise)
        l = kl / torch.sum(z_mask)
        return l, kl_sample_wise

    @staticmethod
    def cosine_similarity_loss(gt_spk_emb, syn_spk_emb):
        l = -torch.nn.functional.cosine_similarity(gt_spk_emb, syn_spk_emb).mean()
        return l

    def forward(
        self,
        waveform,
        waveform_hat,
        z_p,
        logs_q,
        m_p,
        logs_p,
        z_mask,
        scores_disc_fake,
        feats_disc_fake,
        feats_disc_real,
        loss_duration,
        # use_speaker_encoder_as_loss=False,
        # gt_spk_emb=None,
        # syn_spk_emb=None,

        text_lengths=None,
        language_ids=None,
        lang_prediction=None,

        mask=None,
        mel_lengths=None,
        pitch_pred=None,
        pitch_tgt=None,
        energy_pred=None,
        energy_tgt=None,
        # mel_gt=None,
        # mel_gen=None,
        # y_mask=None,
        pitch_flow=None,
        energy_flow=None,
        z_p_pitch_pred=None,
        z_p_energy_pred=None,
        z_p_pitch=None,
        z_p_energy=None,

    ):
        """
        Shapes:
            - waveform : :math:`[B, 1, T]`
            - waveform_hat: :math:`[B, 1, T]`
            - z_p: :math:`[B, C, T]`
            - logs_q: :math:`[B, C, T]`
            - m_p: :math:`[B, C, T]`
            - logs_p: :math:`[B, C, T]`
            - z_len: :math:`[B]`
            - scores_disc_fake[i]: :math:`[B, C]`
            - feats_disc_fake[i][j]: :math:`[B, C, T', P]`
            - feats_disc_real[i][j]: :math:`[B, C, T', P]`
        """

        del pitch_flow, energy_flow, z_p_pitch_pred, z_p_energy_pred, z_p_pitch, z_p_energy, energy_pred, energy_tgt


        loss = 0.0
        loss_mel = torch.tensor([0]).to(waveform)
        loss_gen = torch.tensor([0]).to(waveform)
        loss_feat = torch.tensor([0]).to(waveform)
        loss_pitch = torch.tensor([0]).to(waveform)


        per_sample_kl_loss = torch.tensor([0]).to(waveform)
        per_sample_pitch_loss = torch.tensor([0]).to(waveform)
        per_sample_mel_loss = torch.tensor([0]).to(waveform)

        lang_pred_loss = torch.tensor([0]).float().to(waveform)
        # loss_energy = torch.tensor([0]).float().to(waveform)
        # loss_pitch_cond = torch.tensor([0]).float().to(waveform)
        # loss_energy_cond = torch.tensor([0]).float().to(waveform)
        return_dict = {}


        # The discriminator and hifi losses
        if feats_disc_fake is not None:
            # compute mel spectrograms from the waveforms
            mel = self.stft(waveform)
            mel_hat = self.stft(waveform_hat)
            # loss_mel = torch.nn.functional.l1_loss(mel, mel_hat) * self.mel_loss_alpha
            loss_mel = torch.nn.functional.l1_loss(mel, mel_hat, reduction="none")
            if self.args.analyze_loss:
                per_sample_mel_loss = loss_mel.sum(dim=1).sum(dim=1) * self.mel_loss_alpha
            loss_mel = loss_mel.mean() * self.mel_loss_alpha

            loss_gen = self.generator_loss(scores_disc_fake)[0] * self.gen_loss_alpha
            loss_feat = self.feature_loss(feats_disc_fake, feats_disc_real) * self.feat_loss_alpha

        del waveform, waveform_hat

        if self.args.hifi_only:
            return_dict["loss_gen"] = loss_gen
            return_dict["loss_feat"] = loss_feat
            return_dict["loss_mel"] = loss_mel
            loss = loss_feat + loss_mel + loss_gen
            return_dict["loss"] = loss

            return_dict["loss_kl"] = torch.tensor([0]).float()
            return_dict["loss_duration"] = torch.tensor([0]).float()
            return_dict["loss_pitch"] = torch.tensor([0]).float()
            return_dict["loss_energy"] = torch.tensor([0]).float()
            return return_dict


        # compute other losses
        loss_kl, kl_sample_wise = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask.unsqueeze(1))
        if self.args.analyze_loss:
            per_sample_kl_loss = kl_sample_wise.sum(dim=1).sum(dim=1) * self.kl_loss_alpha
        loss_kl = loss_kl * self.kl_loss_alpha

        loss_duration = torch.sum(loss_duration.float()) * self.dur_loss_alpha

        del z_p, logs_q, m_p, logs_p, scores_disc_fake, feats_disc_fake, feats_disc_real

        # Pitch loss
        if self.args.pitch:
            # Need: dur_lens,pitch_pred,pitch_tgt, max_inp_lengths

            ldiff = pitch_tgt.size(2) - pitch_pred.size(2)

            pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0, 0, 0), value=0.0)
            loss_pitch = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')

            loss_pitch = (loss_pitch * mask.unsqueeze(1))#.sum() / mask.sum()
            if self.args.analyze_loss:
                per_sample_pitch_loss = loss_pitch.sum(dim=1).sum(dim=1).sum(dim=1)



            loss_pitch = loss_pitch.sum() / mask.sum()
            loss_pitch = loss_pitch / pitch_pred.shape[0] # Scale with batch size
            loss_pitch = loss_pitch * self.pitch_predictor_loss_scale

            # if self.args.ow_flow:
            #     pitch_flow = F.pad(pitch_flow, (0, ldiff, 0, 0, 0, 0), value=0.0)

            #     loss_pitch_cond = F.mse_loss(pitch_tgt, pitch_flow, reduction='none')
            #     loss_pitch_cond = (loss_pitch_cond * mask.unsqueeze(1))
            #     loss_pitch_cond = loss_pitch_cond.sum() / mask.sum()
            #     loss_pitch_cond = loss_pitch_cond / pitch_pred.shape[0] # Scale with batch size


        del pitch_pred, pitch_tgt

        if self.args.energy:
            ldiff = energy_tgt.size(2) - energy_pred.size(2)

            energy_pred = F.pad(energy_pred, (0, ldiff, 0, 0), value=0.0)
            loss_energy = F.mse_loss(energy_tgt, energy_pred, reduction='none')
            loss_energy = (loss_energy * mask).sum() / mask.sum()
            loss_energy = loss_energy / energy_pred.shape[0] # Scale with batch size
            loss_energy = loss_energy * self.energy_predictor_loss_scale

            if self.args.ow_flow:
                energy_flow = F.pad(energy_flow, (0, ldiff, 0, 0, 0, 0), value=0.0)

                loss_energy_cond = F.mse_loss(energy_tgt, energy_flow, reduction='none')
                loss_energy_cond = (loss_energy_cond * mask.unsqueeze(1))
                loss_energy_cond = loss_energy_cond.sum() / mask.sum()
                loss_energy_cond = loss_energy_cond / energy_pred.shape[0] # Scale with batch size


        # if self.args.expanded_flow:
        #     loss_pitch_cond = F.mse_loss(z_p_pitch, z_p_pitch_pred, reduction='none')
        #     loss_pitch_cond = (loss_pitch_cond * y_mask.unsqueeze(1)).sum() / y_mask.sum()
        #     loss_pitch_cond = loss_pitch_cond / z_p_pitch_pred.shape[0] # Scale with batch size
        #     loss_pitch_cond = loss_pitch_cond * self.pitch_predictor_loss_scale

        #     if self.args.energy:
        #         loss_energy_cond = F.mse_loss(z_p_energy, z_p_energy_pred, reduction='none')
        #         loss_energy_cond = (loss_energy_cond * y_mask.unsqueeze(1)).sum() / y_mask.sum()
        #         loss_energy_cond = loss_energy_cond / z_p_energy_pred.shape[0] # Scale with batch size
        #         loss_energy_cond = loss_energy_cond * self.energy_predictor_loss_scale


        if self.args.mltts_rc:

            # print(f'text_lengths, {text_lengths}')
            # print(f'mel_lengths, {mel_lengths}')
            # print(f'pitch_pred, {pitch_pred}', pitch_pred.shape)
            # print(f'mask, {mask}', mask.sum(dim=1))
            # text_lengths = torch.tensor(mask.sum(dim=1)).to(lang_prediction)
            text_lengths = mel_lengths
            lang_pred_loss = ReversalClassifier.loss(text_lengths, language_ids, lang_prediction)
            # lang_pred_loss = lang_pred_loss * 10 * 5 #* 25
            # print(f'lang_pred_loss, {lang_pred_loss}')
            # if self.args.mltts_rc_rev:
            #     lang_pred_loss = lang_pred_loss * -1



        loss = loss_kl + loss_feat + loss_mel + loss_gen + loss_duration + loss_pitch + lang_pred_loss# + loss_energy + loss_pitch_cond + loss_energy_cond
        # loss = loss_kl

        # pass losses to the dict
        return_dict["loss_gen"] = loss_gen
        return_dict["loss_kl"] = loss_kl
        return_dict["loss_feat"] = loss_feat
        return_dict["loss_mel"] = loss_mel
        return_dict["loss_duration"] = loss_duration
        return_dict["loss_pitch"] = loss_pitch
        return_dict["lang_pred_loss"] = lang_pred_loss
        # return_dict["loss_energy"] = loss_energy
        # return_dict["loss_pitch_cond"] = loss_pitch_cond
        # return_dict["loss_energy_cond"] = loss_energy_cond
        return_dict["loss"] = loss

        return_dict["per_sample_kl_loss"] = per_sample_kl_loss
        return_dict["per_sample_pitch_loss"] = per_sample_pitch_loss
        return_dict["per_sample_mel_loss"] = per_sample_mel_loss

        return return_dict

class VitsDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc_loss_alpha = 1.0#c.disc_loss_alpha

    @staticmethod
    def discriminator_loss(scores_real, scores_fake):
        loss = 0
        real_losses = []
        fake_losses = []
        for dr, dg in zip(scores_real, scores_fake):
            dr = dr.float()
            dg = dg.float()
            real_loss = torch.mean((1 - dr) ** 2)
            fake_loss = torch.mean(dg ** 2)
            loss += real_loss + fake_loss
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())

        return loss, real_losses, fake_losses

    def forward(self, scores_disc_real, scores_disc_fake):
        loss = 0.0
        return_dict = {}
        loss_disc, _, _ = self.discriminator_loss(scores_disc_real, scores_disc_fake)
        return_dict["loss_disc"] = loss_disc * self.disc_loss_alpha
        loss = loss + return_dict["loss_disc"]
        return_dict["loss"] = loss
        return return_dict


def mask_from_lens(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask