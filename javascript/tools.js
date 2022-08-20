"use strict"

const fs = require("fs-extra")
const {getAudioDurationInSeconds} = require("get-audio-duration")

const lang_names = {
    "am": "Amharic",
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "ha": "Hausa",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "jp": "Japanese",
    "ko": "Korean",
    "la": "Latin",
    "mn": "Mongolian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sw": "Kiswahili",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "yo": "Yoruba",
    "zh": "Chinese Mandarin",
}

const makeLanguageDropdown = () => {
    const languages = fs.readdirSync(`${window.path}/python/transcribe/wav2vec2`)
        .filter(name => !name.startsWith("_")&&!name.includes("."))
        .map(langCode => {return [langCode, lang_names[langCode]]}).sort((a,b) => a[1]<b[1]?-1:1)
    const selectElem = createElem("select")
    languages.forEach(lang => {
        const optionElem = createElem("option", {value: lang[0]})
        optionElem.innerHTML = lang[1]
        selectElem.appendChild(optionElem)
    })
    selectElem.value = "en"
    const langDescription = createElem("div", "Language (Note: Only English training is supported with v2 models. Check v3 models for multi-lingual training.)")
    const rowItemLang = createElem("div", createElem("div", langDescription), createElem("div", selectElem))
    return [rowItemLang, selectElem]
}

const tools = {
    "Audio formatting": {
        taskId: "formatting",
        description: "Audio must be in mono 22050Hz .wav format for all tools and models. This tool will automatically convert your audio to this.",
        inputDirectory: `${window.path}/python/audio_format/input`,
        outputDirectory: `${window.path}/python/audio_format/output`,
        isMultiProcessed: false,
        setupFn: (taskId) => {
            const ckbxDescription = createElem("div", "Use multi-processing")
            const ckbx = createElem("input", {type: "checkbox"})
            ckbx.style.height = "20px"
            ckbx.style.width = "20px"

            window.tools_state.toolSettings["formatting"] = window.tools_state.toolSettings["formatting"] || {}
            window.tools_state.toolSettings["formatting"].useMP = false

            ckbx.addEventListener("click", () => {
                window.tools_state.toolSettings["formatting"].useMP = ckbx.checked
                window.tools_state.toolSettings["formatting"].isMultiProcessed = ckbx.checked
                window.tools_state.isMultiProcessed = ckbx.checked
            })
            const rowItemUseMp = createElem("div", createElem("div", ckbxDescription), createElem("div", ckbx))

            const formattingHzDescription = createElem("div", "Audio Hz (22050 unless you know what you're doing)")
            const formattingHzInput = createElem("input", {type: "number"})
            formattingHzInput.style.width = "70%"
            formattingHzInput.value = "22050"
            formattingHzInput.addEventListener("change", () => {
                window.tools_state.toolSettings["formatting"].formatting_hz = formattingHzInput.value
            })
            const rowItemformattingHzInputs = createElem("div", formattingHzInput)
            rowItemformattingHzInputs.style.flexDirection = "row"
            const rowItemFormattingHz = createElem("div", formattingHzDescription, rowItemformattingHzInputs)
            window.tools_state.toolSettings["formatting"].formatting_hz = formattingHzInput.value


            const container = createElem("div.flexTable.toolSettingsTable", rowItemUseMp, rowItemFormattingHz)
            toolDescription.appendChild(container)
        }
    },
    "AI speaker diarization": {
        taskId: "diarization",
        description: "Speaker diarization can be used to automatically extract slices of speech audio from a really long audio sample, such as audiobooks and movies, based on Speaker Activity Detection. The slices of speech can also be automatically attributed to the correct individual speakers, before being output to file. Long audio files can take a long time to process.",
        inputDirectory: `${window.path}/python/speaker_diarization/input`,
        outputDirectory: `${window.path}/python/speaker_diarization/output/`,
        toolSettings: {},
        inputFileType: ".wav",
        setupFn: (taskId) => {
            window.tools_state.toolSettings["diarization"] = window.tools_state.toolSettings["diarization"] || {}
            window.tools_state.toolSettings["diarization"].mergeSingleOutputFolder = false
            window.tools_state.toolSettings["diarization"].outputAudacityLabels = false

            const ckbxDescription = createElem("div", "Merge output into one folder (Only use when you are sure all files have only one voice)")
            const ckbx = createElem("input", {type: "checkbox"})
            ckbx.style.height = "20px"
            ckbx.style.width = "20px"
            ckbx.addEventListener("click", () => {
                window.tools_state.toolSettings["diarization"].mergeSingleOutputFolder = ckbx.checked
            })
            const container = createElem("div", ckbx, ckbxDescription)
            container.style.display = "flex"
            container.style.justifyContent = "center"
            container.style.alignItems = "center"

            const ckbxAudacityLabelsDescription = createElem("div", "Output labels for Audacity")
            const ckbxAudacityLabels = createElem("input", {type: "checkbox"})
            ckbxAudacityLabels.style.height = "20px"
            ckbxAudacityLabels.style.width = "20px"
            ckbxAudacityLabels.addEventListener("click", () => {
                window.tools_state.toolSettings["diarization"].outputAudacityLabels = ckbxAudacityLabels.checked
            })
            const containerAudacityLabels = createElem("div", ckbxAudacityLabels, ckbxAudacityLabelsDescription)
            containerAudacityLabels.style.display = "flex"
            containerAudacityLabels.style.justifyContent = "center"
            containerAudacityLabels.style.alignItems = "center"

            toolDescription.appendChild(container)
            toolDescription.appendChild(containerAudacityLabels)
        }
    },
    "AI Source separation": {
        taskId: "ass",
        description: "AI audio source separation can be used to extract only the dialogue audio from an audio sample with lots of other audio mixed in, such as music, background sounds, and ambient noises. It can also remove echo from voices. Don't run on long files.",
        inputDirectory: `${window.path}/python/audio_source_separation/input`,
        outputDirectory: `${window.path}/python/audio_source_separation/output/`,
        inputFileType: ".wav"
    },
    "Audio normalization": {
        taskId: "normalize",
        description: "Normalize audio (EBU R128). Use this if your dataset contains audio from multiple sources. Don't use if you got your data straight from game files (only) and you plan to integrate synthesized audio back into the game.",
        inputDirectory: `${window.path}/python/audio_norm/input`,
        outputDirectory: `${window.path}/python/audio_norm/output/`,
        inputFileType: ".wav",
        isMultiProcessed: false,
        setupFn: (taskId) => {
            const ckbxDescription = createElem("div", "Use multi-processing")
            const ckbx = createElem("input", {type: "checkbox"})
            ckbx.style.height = "20px"
            ckbx.style.width = "20px"

            window.tools_state.toolSettings["normalize"] = window.tools_state.toolSettings["normalize"] || {}
            window.tools_state.toolSettings["normalize"].useMP = false

            ckbx.addEventListener("click", () => {
                window.tools_state.toolSettings["normalize"].useMP = ckbx.checked
                window.tools_state.toolSettings["normalize"].isMultiProcessed = ckbx.checked
                window.tools_state.isMultiProcessed = ckbx.checked
            })

            const rowItemUseMp = createElem("div", createElem("div", ckbxDescription), createElem("div", ckbx))

            const normalizationHzDescription = createElem("div", "Audio Hz (22050 unless you know what you're doing)")
            const normalizationHzInput = createElem("input", {type: "number"})
            normalizationHzInput.style.width = "70%"
            normalizationHzInput.value = "22050"
            normalizationHzInput.addEventListener("change", () => {
                window.tools_state.toolSettings["normalize"].normalization_hz = normalizationHzInput.value
            })
            const rowItemNormalizationHzInputs = createElem("div", normalizationHzInput)
            rowItemNormalizationHzInputs.style.flexDirection = "row"
            const rowItemNormalizationHz = createElem("div", normalizationHzDescription, rowItemNormalizationHzInputs)
            window.tools_state.toolSettings["normalize"].normalization_hz = normalizationHzInput.value


            const container = createElem("div.flexTable.toolSettingsTable", rowItemUseMp, rowItemNormalizationHz)
            toolDescription.appendChild(container)
        }
    },
    "WEM to OGG": {
        taskId: "wem2ogg",
        description: "Convert .wem files to .ogg files. Use the audio formatting tool to then convert the .ogg files to the .wav format required by tools/models.",
        inputDirectory: `${window.path}/python/wem2ogg/input`,
        outputDirectory: `${window.path}/python/wem2ogg/output/`,
        inputFileType: ".wem"
    },
    "Make .srt": {
        taskId: "make_srt",
        description: "Create an .srt for mp4 files, using the speaker diarization and auto-transcript tools. Useful for sanity checking, and easier external tweaking when using data from a video.",
        inputDirectory: `${window.path}/python/make_srt/input`,
        outputDirectory: `${window.path}/python/make_srt/output/`,
        inputFileType: "folder",
        setupFn: (taskId) => {
            window.tools_state.toolSettings["make_srt"] = window.tools_state.toolSettings["make_srt"] || {}
            window.tools_state.toolSettings["make_srt"].language = "en"

            const [rowItemLang, selectElem] = makeLanguageDropdown()
            selectElem.addEventListener("change", () => window.tools_state.toolSettings["make_srt"].language=selectElem.value)

            const container = createElem("div.flexTable.toolSettingsTable", rowItemLang)
            toolDescription.appendChild(container)
        }
    },
    "Cluster speakers": {
        taskId: "cluster_speakers",
        description: "Group up audio files of one or more speakers into several clusters based on speaking style/identity. Aim for max ~10k files at a time.",
        inputDirectory: `${window.path}/python/cluster_speakers/input`,
        outputDirectory: `${window.path}/python/cluster_speakers/output/`,
        inputFileType: "folder",
        setupFn: (taskId) => {

            window.tools_state.toolSettings["cluster_speakers"] = window.tools_state.toolSettings["cluster_speakers"] || {}
            window.tools_state.toolSettings["cluster_speakers"].use_custom_k = false
            window.tools_state.toolSettings["cluster_speakers"].custom_k = false
            window.tools_state.toolSettings["cluster_speakers"].use_min_cluster_size = false
            window.tools_state.toolSettings["cluster_speakers"].min_cluster_size = false
            window.tools_state.toolSettings["cluster_speakers"].do_search_reordering = false
            window.tools_state.toolSettings["cluster_speakers"].use_cluster_folder_prefix = false
            window.tools_state.toolSettings["cluster_speakers"].cluster_folder_prefix = "0001"
            window.tools_state.isMultiProcessed = false // ?

            const doSearchReorderCkbxDescription = createElem("div", "Re-order clusters by similarity to the biggest (principal) cluster")
            const doSearchReorderCkbx = createElem("input", {type: "checkbox"})
            doSearchReorderCkbx.style.height = "20px"
            doSearchReorderCkbx.style.width = "20px"
            doSearchReorderCkbx.addEventListener("click", () => {
                window.tools_state.toolSettings["cluster_speakers"].do_search_reordering = doSearchReorderCkbx.checked
            })
            const rowItemReorder = createElem("div", doSearchReorderCkbxDescription, createElem("div", doSearchReorderCkbx))

            const fixedKCkbx = createElem("input", {type: "checkbox"})
            fixedKCkbx.style.height = "20px"
            fixedKCkbx.style.width = "20px"
            fixedKCkbx.addEventListener("click", () => {
                window.tools_state.toolSettings["cluster_speakers"].use_custom_k = fixedKCkbx.checked
                inputFixedKValInput.disabled = !fixedKCkbx.checked
            })

            const inputFixedKValDescription = createElem("div", "Use custom fixed number of clusters")
            const inputFixedKValInput = createElem("input", {type: "number"})
            inputFixedKValInput.disabled = true
            inputFixedKValInput.value = 2
            inputFixedKValInput.style.width = "70%"
            inputFixedKValInput.addEventListener("change", () => {
                window.tools_state.toolSettings["cluster_speakers"].custom_k = parseInt(inputFixedKValInput.value)
            })
            const rowItemFixedKAmountInputs = createElem("div", fixedKCkbx, inputFixedKValInput)
            rowItemFixedKAmountInputs.style.flexDirection = "row"
            const rowItemFixedKAmount = createElem("div", inputFixedKValDescription, rowItemFixedKAmountInputs)
            window.tools_state.toolSettings["cluster_speakers"].custom_k = parseInt(inputFixedKValInput.value)


            const useMinCSizeCkbx = createElem("input", {type: "checkbox"})
            useMinCSizeCkbx.style.height = "20px"
            useMinCSizeCkbx.style.width = "20px"
            useMinCSizeCkbx.addEventListener("click", () => {
                window.tools_state.toolSettings["cluster_speakers"].use_min_cluster_size = useMinCSizeCkbx.checked
                minCSizeInput.disabled = !useMinCSizeCkbx.checked
            })

            const minCSizeDescription = createElem("div", "Only output clusters with a minimum number of files in it")
            const minCSizeInput = createElem("input", {type: "number"})
            minCSizeInput.disabled = true
            minCSizeInput.value = 2
            minCSizeInput.style.width = "70%"
            minCSizeInput.addEventListener("change", () => {
                window.tools_state.toolSettings["cluster_speakers"].min_cluster_size = parseInt(minCSizeInput.value)
            })
            const rowItemminCSizeInputs = createElem("div", useMinCSizeCkbx, minCSizeInput)
            rowItemminCSizeInputs.style.flexDirection = "row"
            const rowItemminCSize = createElem("div", minCSizeDescription, rowItemminCSizeInputs)
            window.tools_state.toolSettings["cluster_speakers"].min_cluster_size = parseInt(minCSizeInput.value)

            const useClusterFoldersPrefix = createElem("input", {type: "checkbox"})
            useClusterFoldersPrefix.style.height = "20px"
            useClusterFoldersPrefix.style.width = "20px"
            useClusterFoldersPrefix.addEventListener("click", () => {
                window.tools_state.toolSettings["cluster_speakers"].use_cluster_folder_prefix = useClusterFoldersPrefix.checked
                clusterFolderPrefixInput.disabled = !useClusterFoldersPrefix.checked
            })

            const clusterFolderPrefixDescription = createElem("div", "Prefix cluster folder names with something")
            const clusterFolderPrefixInput = createElem("input")
            clusterFolderPrefixInput.disabled = true
            clusterFolderPrefixInput.style.width = "70%"
            clusterFolderPrefixInput.value = "0001"
            clusterFolderPrefixInput.addEventListener("change", () => {
                window.tools_state.toolSettings["cluster_speakers"].cluster_folder_prefix = clusterFolderPrefixInput.value
            })
            const rowItemclusterFolderPrefixInputs = createElem("div", useClusterFoldersPrefix, clusterFolderPrefixInput)
            rowItemclusterFolderPrefixInputs.style.flexDirection = "row"
            const rowItemclusterFolderPrefix = createElem("div", clusterFolderPrefixDescription, rowItemclusterFolderPrefixInputs)
            window.tools_state.toolSettings["cluster_speakers"].min_cluster_size = clusterFolderPrefixInput.value

            const container = createElem("div.flexTable.toolSettingsTable", rowItemReorder, rowItemFixedKAmount, rowItemminCSize, rowItemclusterFolderPrefix)
            toolDescription.appendChild(container)
        }
    },
    "Speaker similarity search": {
        taskId: "speaker_search",
        description: "Use one or more speaker audio files as a query to search a big corpus of speaker audio files for similarity. The corpus is copied into the output folder with numerically incremental prefix filenames, such that the files are sorted by similarity to your search query.",
        inputDirectory: `${window.path}/python/speaker_search/query`,
        inputDirectory2: `${window.path}/python/speaker_search/corpus`,
        outputDirectory: `${window.path}/python/speaker_search/results/`,
        inputFileType: "folder"
    },
    "Speaker cluster similarity search": {
        taskId: "speaker_cluster_search",
        description: "Use one or more speaker audio files as a query to search a larger corpus of data, sorting whole groups (clusters) as results at a time. Same as speaker search, but treating each folder of audio in the input folder (same structure as the output of the clustering tool) as a data item, before copying it into the ouput directory.",
        inputDirectory: `${window.path}/python/speaker_cluster_search/query`,
        inputDirectory2: `${window.path}/python/speaker_cluster_search/corpus`,
        outputDirectory: `${window.path}/python/speaker_cluster_search/results/`,
        inputFileType: "folder"
    },
    "Transcribe": {
        taskId: "transcribe",
        description: "Automatically generate a transcript for the speaker audio files in the input folder. This is the same tool that is available on the main screen.",
        inputDirectory: `${window.path}/python/transcribe/input`,
        outputDirectory: `${window.path}/python/transcribe/output/`,
        inputFileType: "folder",
        isMultiProcessed: false,
        setupFn: (taskId) => {
            window.tools_state.toolSettings["transcribe"] = window.tools_state.toolSettings["transcribe"] || {}
            window.tools_state.toolSettings["transcribe"].ignore_existing_transcript = false
            window.tools_state.toolSettings["transcribe"].language = "en"

            const [rowItemLang, selectElem] = makeLanguageDropdown()
            selectElem.addEventListener("change", () => window.tools_state.toolSettings["transcribe"].language=selectElem.value)


            const ckbxDescription = createElem("div", "Use multi-processing")
            const ckbx = createElem("input", {type: "checkbox"})
            ckbx.style.height = "20px"
            ckbx.style.width = "20px"

            window.tools_state.toolSettings["transcribe"] = window.tools_state.toolSettings["transcribe"] || {}
            window.tools_state.toolSettings["transcribe"].useMP = false

            ckbx.addEventListener("click", () => {
                window.tools_state.toolSettings["transcribe"].useMP = ckbx.checked
            })
            const rowItemUseMp = createElem("div", createElem("div", ckbxDescription), createElem("div", ckbx))


            const transcribeMPworkersDescription = createElem("div", "Number of processes (Set low - quite RAM intense)")
            const transcribeMPworkersInput = createElem("input", {type: "number"})
            transcribeMPworkersInput.style.width = "70%"
            transcribeMPworkersInput.value = "2"
            transcribeMPworkersInput.addEventListener("change", () => {
                window.tools_state.toolSettings["transcribe"].useMP_num_workers = transcribeMPworkersInput.value
            })
            const rowItemtranscribeMPworkersInputs = createElem("div", transcribeMPworkersInput)
            rowItemtranscribeMPworkersInputs.style.flexDirection = "row"
            const rowItemtranscribeMPworkers = createElem("div", transcribeMPworkersDescription, rowItemtranscribeMPworkersInputs)
            window.tools_state.toolSettings["transcribe"].useMP_num_workers = transcribeMPworkersInput.value



            const container = createElem("div.flexTable.toolSettingsTable", rowItemUseMp, rowItemtranscribeMPworkers, rowItemLang)
            toolDescription.appendChild(container)
        }
    },
    "WER transcript evaluation": {
        taskId: "wer_evaluation",
        description: "Evaluate the quality of a transcript file's text against the text auto-generated by the Transcription tool.",
        inputDirectory: `${window.path}/python/wer_evaluation/input`,
        inputDirectory2: `${window.path}/python/wer_evaluation/asr_reference`,
        outputDirectory: `${window.path}/python/wer_evaluation/output/`,
        inputFileType: "folder"
    },
    "Remove background noise": {
        taskId: "noise_removal",
        description: "Automatically remove constant background hiss/hum/noise from audio, using a reference 'silent' clip with just the background noise.",
        inputDirectory: `${window.path}/python/noise_removal/input`,
        inputDirectory2: `${window.path}/python/noise_removal/noise`,
        outputDirectory: `${window.path}/python/noise_removal/output/`,
        inputFileType: "folder"
    },
    "Cut padding": {
        taskId: "cut_padding",
        description: "Remove starting and ending silence from clips, based on configurable silence detection.",
        inputDirectory: `${window.path}/python/cut_padding/input`,
        outputDirectory: `${window.path}/python/cut_padding/output/`,
        inputFileType: ".wav",
        isMultiProcessed: false,
        setupFn: (taskId) => {
            const ckbxDescription = createElem("div", "Use multi-processing")
            const ckbx = createElem("input", {type: "checkbox"})
            ckbx.style.height = "20px"
            ckbx.style.width = "20px"

            window.tools_state.toolSettings["cut_padding"] = window.tools_state.toolSettings["cut_padding"] || {}
            window.tools_state.toolSettings["cut_padding"].useMP = false

            ckbx.addEventListener("click", () => {
                window.tools_state.toolSettings["cut_padding"].useMP = ckbx.checked
                window.tools_state.toolSettings["cut_padding"].isMultiProcessed = ckbx.checked
                window.tools_state.isMultiProcessed = ckbx.checked
            })

            const rowItemUseMp = createElem("div", createElem("div", ckbxDescription), createElem("div", ckbx))

            const cutPaddingSilenceThresholdDescription = createElem("div", "Silence threshold (dB)")
            const cutPaddingSilenceThresholdInput = createElem("input", {type: "number"})
            cutPaddingSilenceThresholdInput.style.width = "70%"
            cutPaddingSilenceThresholdInput.value = "-65"
            cutPaddingSilenceThresholdInput.addEventListener("change", () => {
                window.tools_state.toolSettings["cut_padding"].min_dB = cutPaddingSilenceThresholdInput.value
            })
            const rowItemcutPaddingSilenceThresholdInputs = createElem("div", cutPaddingSilenceThresholdInput)
            rowItemcutPaddingSilenceThresholdInputs.style.flexDirection = "row"
            const rowItemcutPaddingSilenceThreshold = createElem("div", cutPaddingSilenceThresholdDescription, rowItemcutPaddingSilenceThresholdInputs)
            window.tools_state.toolSettings["cut_padding"].min_dB = cutPaddingSilenceThresholdInput.value


            const container = createElem("div.flexTable.toolSettingsTable", rowItemUseMp, rowItemcutPaddingSilenceThreshold)
            toolDescription.appendChild(container)
        }
    },
    "Silence split": {
        taskId: "silence_split",
        description: "Split audio based on configurable silence detection. Use the background noise removal tool if there's a constant hiss/hum/noise throughout your clips.",
        inputDirectory: `${window.path}/python/silence_split/input`,
        outputDirectory: `${window.path}/python/silence_split/output/`,
        inputFileType: ".wav",
        isMultiProcessed: false,
        setupFn: (taskId) => {
            const ckbxDescription = createElem("div", "Use multi-processing")
            const ckbx = createElem("input", {type: "checkbox"})
            ckbx.style.height = "20px"
            ckbx.style.width = "20px"

            window.tools_state.toolSettings["silence_split"] = window.tools_state.toolSettings["silence_split"] || {}
            window.tools_state.toolSettings["silence_split"].useMP = false

            ckbx.addEventListener("click", () => {
                window.tools_state.toolSettings["silence_split"].useMP = ckbx.checked
                window.tools_state.toolSettings["silence_split"].isMultiProcessed = ckbx.checked
                window.tools_state.isMultiProcessed = ckbx.checked
            })

            const rowItemUseMp = createElem("div", createElem("div", ckbxDescription), createElem("div", ckbx))

            const silenceSplitSilenceThresholdDescription = createElem("div", "Silence threshold (dB)")
            const silenceSplitSilenceThresholdInput = createElem("input", {type: "number"})
            silenceSplitSilenceThresholdInput.style.width = "70%"
            silenceSplitSilenceThresholdInput.value = "-10"
            silenceSplitSilenceThresholdInput.addEventListener("change", () => {
                window.tools_state.toolSettings["silence_split"].min_dB = silenceSplitSilenceThresholdInput.value
            })
            const rowItemsilenceSplitSilenceThresholdInputs = createElem("div", silenceSplitSilenceThresholdInput)
            rowItemsilenceSplitSilenceThresholdInputs.style.flexDirection = "row"
            const rowItemSilenceSplitSilenceThreshold = createElem("div", silenceSplitSilenceThresholdDescription, rowItemsilenceSplitSilenceThresholdInputs)
            window.tools_state.toolSettings["silence_split"].min_dB = silenceSplitSilenceThresholdInput.value


            const silenceSplitSilenceDurationDescription = createElem("div", "Silence duration (s)")
            const silenceSplitSilenceDurationInput = createElem("input", {type: "number"})
            silenceSplitSilenceDurationInput.style.width = "70%"
            silenceSplitSilenceDurationInput.value = "0.25"
            silenceSplitSilenceDurationInput.addEventListener("change", () => {
                window.tools_state.toolSettings["silence_split"].silence_duration = silenceSplitSilenceDurationInput.value
            })
            const rowItemsilenceSplitSilenceDurationInputs = createElem("div", silenceSplitSilenceDurationInput)
            rowItemsilenceSplitSilenceDurationInputs.style.flexDirection = "row"
            const rowItemSilenceSplitSilenceDuration = createElem("div", silenceSplitSilenceDurationDescription, rowItemsilenceSplitSilenceDurationInputs)
            window.tools_state.toolSettings["silence_split"].silence_duration = silenceSplitSilenceDurationInput.value


            // const container = createElem("div.flexTable.toolSettingsTable", rowItemUseMp, rowItemSilenceSplitSilenceThreshold, rowItemSilenceSplitSilenceDuration)
            const container = createElem("div.flexTable.toolSettingsTable", rowItemSilenceSplitSilenceThreshold, rowItemSilenceSplitSilenceDuration)
            toolDescription.appendChild(container)
        }
    },
    "SRT split": {
        taskId: "srt_split",
        description: "Split long .mp4 clips based on their accompanying .srt subtitle file, into short clips with the associated transcript.",
        inputDirectory: `${window.path}/python/srt_split/input`,
        outputDirectory: `${window.path}/python/srt_split/output`,
        inputFileType: "folder",
        isMultiProcessed: false,
        setupFn: (taskId) => {
            const ckbxDescription = createElem("div", "Use multi-processing")
            const ckbx = createElem("input", {type: "checkbox"})
            ckbx.style.height = "20px"
            ckbx.style.width = "20px"

            window.tools_state.toolSettings["srt_split"] = window.tools_state.toolSettings["srt_split"] || {}
            window.tools_state.toolSettings["srt_split"].useMP = false

            ckbx.addEventListener("click", () => {
                window.tools_state.toolSettings["srt_split"].useMP = ckbx.checked
                window.tools_state.toolSettings["srt_split"].isMultiProcessed = ckbx.checked
                window.tools_state.isMultiProcessed = ckbx.checked
            })
            const rowItemUseMp = createElem("div", createElem("div", ckbxDescription), createElem("div", ckbx))

            const container = createElem("div.flexTable.toolSettingsTable", rowItemUseMp)
            toolDescription.appendChild(container)
        }
    },
}


// Brute force progress indicator, for when the WebSockets don't work
setInterval(() => {
    if (["transcribe", "srt_split", "make_srt"].includes(window.tools_state.taskId)) {
        if (window.tools_state.running && window.tools_state.taskId && fs.existsSync(`${window.path}/python/${window.tools_state.taskId}/.progress.txt`)) {
            const fileData = fs.readFileSync(`${window.path}/python/${window.tools_state.taskId}/.progress.txt`, "utf8")
            if (fileData.length) {
                toolProgressInfo.innerHTML = fileData
            }
        } else {
            toolProgressInfo.innerHTML = ""
        }
    }

}, 1000)


window.tools_state = {
    running: false,
    inputDirectory: null,
    outputDirectory: null,
    taskFileIndex: 0,
    taskFiles: [],
    taskId: undefined,
    inputFileType: undefined,
    spinnerElem: toolsSpinner,
    progressElem: toolProgressInfo,
    infoElem: toolsInfo,
    currentFileElem: toolsCurrentFile,

    toolSettings: {}
}


Object.keys(tools).sort().forEach(toolName => {
    const button = createElem("button", toolName)
    button.addEventListener("click", () => {

        toolContent.style.display = "block"
        const tool = tools[toolName]

        toolNameElem.innerHTML = toolName
        toolDescription.innerHTML = tool.description

        window.tools_state.taskId = tool.taskId
        window.tools_state.inputDirectory = tool.inputDirectory
        window.tools_state.inputDirectory2 = tool.inputDirectory2
        window.tools_state.outputDirectory = tool.outputDirectory
        window.tools_state.inputFileType = tool.inputFileType

        window.tools_state.spinnerElem = toolsSpinner
        window.tools_state.progressElem = toolProgressInfo
        window.tools_state.infoElem = toolsInfo
        window.tools_state.currentFileElem = toolsCurrentFile
        window.tools_state.isMultiProcessed = false

        toolProgressInfo.innerHTML = ""

        try {
            fs.unlinkSync(`${window.path}/python/${window.tools_state.taskId}/.progress.txt`)
        } catch (e) {}

        if (tool.inputDirectory2) {
            toolsOpenInput2.style.display = "inline-block"
        } else {
            toolsOpenInput2.style.display = "none"
        }

        if (tool.setupFn) {
            tool.setupFn(tool.taskId)
        }
    })
    toolsList.appendChild(button)
})
toolsOpenInput.addEventListener("click", () => {
    // Until I finally upgrade Electron version, to gain access to the function that implicitly does this
    if (!fs.existsSync(window.tools_state.inputDirectory)) {
        fs.mkdirSync(window.tools_state.inputDirectory)
    }
    const files = fs.readdirSync(window.tools_state.inputDirectory)
    if (files.length) {
        shell.showItemInFolder(`${window.tools_state.inputDirectory}/${files[0]}`)
    } else {
        shell.showItemInFolder(`${window.tools_state.inputDirectory}`)
    }
})
toolsOpenInput2.addEventListener("click", () => {
    if (window.tools_state.inputDirectory2) {
        if (!fs.existsSync(window.tools_state.inputDirectory2)) {
            fs.mkdirSync(window.tools_state.inputDirectory2)
        }
        // Until I finally upgrade Electron version, to gain access to the function that implicitly does this
        const files = fs.readdirSync(window.tools_state.inputDirectory2)
        if (files.length) {
            shell.showItemInFolder(`${window.tools_state.inputDirectory2}/${files[0]}`)
        } else {
            shell.showItemInFolder(`${window.tools_state.inputDirectory2}`)
        }
    }
})
toolsOpenOutput.addEventListener("click", () => {
    // Until I finally upgrade Electron version, to gain access to the function that implicitly does this
    if (!fs.existsSync(window.tools_state.outputDirectory)) {
        fs.mkdirSync(window.tools_state.outputDirectory)
    }
    const files = fs.readdirSync(window.tools_state.outputDirectory)
    if (files.length) {
        shell.showItemInFolder(`${window.tools_state.outputDirectory}/${files[0]}`)
    } else {
        shell.showItemInFolder(`${window.tools_state.outputDirectory}`)
    }
})
toolsRunTool.addEventListener("click", () => {

    window.tools_state.taskFileIndex = 0
    window.tools_state.taskFiles = []
    window.tools_state.progressElem.innerHTML = ""

    window.tools_state.taskFiles = fs.readdirSync(window.tools_state.inputDirectory).filter(f => !f.endsWith(".ini"))
    if (!window.tools_state.taskFiles.length) {
        return window.errorModal(`There are no files in the tool's input directory: <br>${window.tools_state.inputDirectory}`)
    }

    if (window.tools_state.spinnerElem) {
        window.tools_state.spinnerElem.style.display = "inline-block"
    }
    toolsRunTool.disabled = true
    prepAudioStart.disabled = true
    window.tools_state.running = true
    toolsList.querySelectorAll("button").forEach(button => button.disabled = true)
    window.deleteFolderRecursive(window.tools_state.outputDirectory, true)


    if (window.tools_state.isMultiProcessed) {
        if (window.tools_state.inputFileType) {
            window.tools_state.taskFiles = window.tools_state.taskFiles.filter(f => f.endsWith(window.tools_state.inputFileType))
        }
        // window.tools_state.inputFileType = "folder"
        if (!fs.existsSync(window.tools_state.outputDirectory)) {
            fs.mkdirSync(window.tools_state.outputDirectory)
        }
        setTimeout(updateMPProgress, 1000)

        const toolSettings = window.tools_state.toolSettings[window.tools_state.taskId] || {}
        window.ws.send(JSON.stringify({model: window.tools_state.taskId, task: "runTask", data: {
            outputDirectory: window.tools_state.outputDirectory,
            toolSettings,
            inPath: window.tools_state.inputDirectory,
            inPath2: window.tools_state.inputDirectory2,
        }}))

    } else {
        if (window.tools_state.inputFileType=="folder") {
            window.tools_state.taskFiles = [""]

        } else if (window.tools_state.inputFileType) {
            window.tools_state.taskFiles = window.tools_state.taskFiles.filter(f => f.endsWith(window.tools_state.inputFileType))
        }

        doNextTaskItem()
    }
})

const doNextTaskItem = () => {
    const inPath = `${window.tools_state.inputDirectory}/${window.tools_state.taskFiles[window.tools_state.taskFileIndex]}`
    const inPath2 = window.tools_state.inputDirectory2

    if (!window.tools_state.taskFiles.length) {
        return window.errorModal("No input files of the required type were found.").then(() => {
            if (window.tools_state.spinnerElem) {
                window.tools_state.spinnerElem.style.display = "none"
            }
            window.tools_state.progressElem.innerHTML = ""
            toolProgressInfo.innerHTML = ""
            toolsRunTool.disabled = false
            prepAudioStart.disabled = false
            toolsList.querySelectorAll("button").forEach(button => button.disabled = false)
            window.tools_state.infoElem.innerHTML = ""
            window.tools_state.currentFileElem.innerHTML = ""
        })
    }

    if (window.tools_state.taskFiles[window.tools_state.taskFileIndex].length) {
        window.tools_state.currentFileElem.innerHTML = `File: ${window.tools_state.taskFiles[window.tools_state.taskFileIndex]}`

        let percentDone = parseInt(window.tools_state.taskFileIndex / window.tools_state.taskFiles.length * 100 * 100) / 100
        toolProgressInfo.innerHTML = `${percentDone}%`

    } else {
        window.tools_state.currentFileElem.innerHTML = ""
    }

    // const toolMeta = Object.keys(tools).find(key => tools[key].taskId==tools_state.taskId)
    // const toolSettings = toolMeta.toolSettings || {}
    const toolSettings = window.tools_state.toolSettings[window.tools_state.taskId] || {}


    window.ws.send(JSON.stringify({model: window.tools_state.taskId, task: "runTask", data: {
        outputDirectory: window.tools_state.outputDirectory,
        toolSettings,
        inPath2,
        inPath
    }}))
}

const updateMPProgress = () => {

    const outputFiles = fs.readdirSync(window.tools_state.outputDirectory)
    const filesDone = outputFiles.filter(fname => window.tools_state.taskFiles.map(fname => fname.split(".").reverse().slice(1,1000).reverse().join(".")+".wav").includes(fname)).length

    if (!fs.existsSync(`${window.path}/python/${window.tools_state.taskId}/.progress.txt`)) {
        const percentDone = parseInt(filesDone/window.tools_state.taskFiles.length*100*100)/100
        window.tools_state.currentFileElem.innerHTML = `Files done: ${filesDone}/${window.tools_state.taskFiles.length} (${percentDone}%)`
    }

    if (filesDone==window.tools_state.taskFiles.length && !fs.existsSync(`${window.path}/python/${window.tools_state.taskId}/.progress.txt`)) {
        window.tools_state.taskFiles = []
        window.tools_state.currentFileElem.innerHTML = ""
        console.log("about to force call tasks_next")
        window.websocket_handlers["tasks_next"](undefined, true)
    } else {
        setTimeout(updateMPProgress, 1000)
    }
}


window.websocket_handlers["task_info"] = (data) => {
    window.tools_state.infoElem.innerHTML = data
}
window.websocket_handlers["tasks_error"] = (data) => {

    console.log("tasks_error", data)
    window.errorModal(data)

    if (window.tools_state.spinnerElem) {
        window.tools_state.spinnerElem.style.display = "none"
    }
    window.tools_state.progressElem.innerHTML = ""
    toolsRunTool.disabled = false
    prepAudioStart.disabled = false
    window.tools_state.infoElem.innerHTML = ""
    window.tools_state.currentFileElem.innerHTML = ""
}

window.websocket_handlers["tasks_next"] = (data, mpOverride=false) => {

    if (window.tools_state.isMultiProcessed && !mpOverride) {
        return
    }

    window.tools_state.progressElem.innerHTML = `${window.tools_state.taskFileIndex}/${window.tools_state.taskFiles.length} files done (${parseInt(window.tools_state.taskFileIndex/window.tools_state.taskFiles.length*100*100)/100}%)`

    if (window.tools_state.taskFileIndex<window.tools_state.taskFiles.length-1) {
        window.tools_state.taskFileIndex++
        doNextTaskItem()
    } else {
        if (window.tools_state.spinnerElem) {
            window.tools_state.spinnerElem.style.display = "none"
        }
        window.tools_state.progressElem.innerHTML = "Done"
        toolsRunTool.disabled = false
        prepAudioStart.disabled = false
        toolsList.querySelectorAll("button").forEach(button => button.disabled = false)
        window.tools_state.infoElem.innerHTML = ""
        window.tools_state.currentFileElem.innerHTML = ""

        if (window.tools_state.post_callback) {
            window.tools_state.post_callback()
        }
    }
}




const runAudioConversionForDataset = (fromDir, toDir) => {
    return new Promise((resolve) => {
        window.tools_state.taskId = "formatting"
        window.tools_state.inputDirectory = fromDir
        window.tools_state.outputDirectory = toDir
        window.tools_state.inputFileType = undefined

        window.tools_state.spinnerElem = undefined//prepAudioSpinner
        window.tools_state.progressElem = prepAudioProgress
        window.tools_state.infoElem = prepAudioCurrentTask
        window.tools_state.currentFileElem = prepAudioCurrentFile

        window.tools_state.post_callback = () => {
            resolve()
        }

        toolsRunTool.click()
    })
}
const runAudioSilenceCutForDataset = (fromDir, toDir) => {
    // TODO - update the UI with a progress meter, like with the tools' mp progress
    return new Promise((resolve) => {
        // TODO

        window.tools_state.taskId = "silence_cut"
        window.tools_state.inputDirectory = fromDir
        window.tools_state.outputDirectory = toDir
        window.tools_state.inputFileType = undefined

        window.tools_state.spinnerElem = undefined//prepAudioSpinner
        window.tools_state.progressElem = prepAudioProgress
        window.tools_state.infoElem = prepAudioCurrentTask
        window.tools_state.currentFileElem = prepAudioCurrentFile


        window.tools_state.post_callback = () => {
            resolve()
        }

        window.ws.send(JSON.stringify({model: window.tools_state.taskId, task: "runTask", data: {
            inputDirectory: window.tools_state.inputDirectory,
            outputDirectory: window.tools_state.outputDirectory
        }}))
    })
}

const timeoutSleep = ms => {
    return new Promise(resolve => {
        setTimeout(() => resolve(), ms)
    })
}

// Preprocess audio
prepAudioStart.addEventListener("click", () => {

    window.confirmModal("Run audio pre-processing?").then(async resp => {
        if (resp) {
            // Disable the tool buttons, to prevent parallel execution of other tools/tool instances
            Array.from(toolsList.querySelectorAll("button")).forEach(button => {
                button.disabled = true
            })

            window.tools_state.running = true

            try {
                // Copy the wavs directory to wavs_backup if doing conversion, or wavs_orig otherwise
                // Run the conversion task for the audio, if turned on, from wavs_backup to wavs_orig if doing silence cutting, or wavs otherwise
                // Run the silence cutting for the audio, if turned on, from wavs_orig to wavs
                // Remove short files from wavs, if enabled
                // Remove long files from wavs, if enabled
                // if not doing backups, delete wavs_backup if doing conversion, and delete wavs_orig
                // else rename wavs_orig to wavs_backup if not doing conversion, else delete wavs_orig

                // ============

                const DOING_CONVERSION = prepAudioConvert.checked
                const DOING_SILENCE_CUT = prepAudioTrimSilence.checked
                const DELETING_SHORT = prepAudioDelShort.checked
                const DELETING_LONG = prepAudioDelLong.checked
                const DOING_BACKUP = prepAudioBackup.checked

                prepAudioSpinner.style.display = "inline-block"


                // Copy the wavs directory to wavs_backup if doing conversion, or wavs_orig otherwise
                prepAudioCurrentTask.innerHTML = "Copying out original data..."

                await timeoutSleep(10) // Give the UI a chance to refresh, to display the info before starting

                if (DOING_CONVERSION) {
                    if (fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_backup`)) {
                        window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_backup`)
                    }
                    fs.copySync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_backup`)
                    // window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`, true)
                } else {
                    if (fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)) {
                        window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)
                    }
                    fs.copySync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)
                    // window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`, true)
                }
                window.tools_state.progressElem.innerHTML = ""

                // Run the conversion task for the audio, if turned on, from wavs_backup to wavs_orig if doing silence cutting, or wavs otherwise
                if (DOING_CONVERSION) {
                    if (DOING_SILENCE_CUT) {
                        if (fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)) {
                            window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)
                        }
                        fs.mkdirSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)
                        prepAudioCurrentTask.innerHTML = "Running audio conversion..."
                        await runAudioConversionForDataset(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_backup`, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)
                    } else {
                        prepAudioCurrentTask.innerHTML = "Running audio conversion..."
                        await runAudioConversionForDataset(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_backup`, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`)
                    }
                }
                window.tools_state.progressElem.innerHTML = ""

                // Run the silence cutting for the audio, if turned on, from wavs_orig to wavs
                if (DOING_SILENCE_CUT) {
                    prepAudioCurrentTask.innerHTML = "Running silence cutting..."
                    await runAudioSilenceCutForDataset(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`)
                }
                window.tools_state.progressElem.innerHTML = ""


                // Remove short files from wavs, if enabled
                // Remove long files from wavs, if enabled
                if (DELETING_SHORT || DELETING_LONG) {

                    prepAudioCurrentTask.innerHTML = "Removing files too "+ (DELETING_SHORT ? (DELETING_LONG ? "long/short" : "short") : "long") + "..."
                    const audioFiles = fs.readdirSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`)

                    for (let ai=0; ai<audioFiles.length; ai++) {
                        const filePath = `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${audioFiles[ai]}`
                        const duration = await getAudioDurationInSeconds(filePath)

                        prepAudioCurrentTask.innerHTML = "Removing files too "+ (DELETING_SHORT ? (DELETING_LONG ? "long/short" : "short") : "long") + `... (${ai+1}/${audioFiles.length})`

                        if (DELETING_SHORT && duration<1) {
                            fs.unlinkSync(filePath)
                        } else if (DELETING_LONG && duration>10) {
                            fs.unlinkSync(filePath)
                        }
                    }
                }

                prepAudioCurrentTask.innerHTML = "Cleaning up..."

                // if not doing backups, delete wavs_backup if doing conversion, and delete wavs_orig
                if (!DOING_BACKUP) {
                    if (DOING_CONVERSION) {
                        window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_backup`)
                    }
                    window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)
                } else {
                    // else rename wavs_orig to wavs_backup if not doing conversion, else delete wavs_orig
                    if (!DOING_CONVERSION) {
                        fs.copySync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_backup`)
                        window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)
                    } else {
                        window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs_orig`)
                    }
                }
                prepAudioCurrentTask.innerHTML = ""
                window.tools_state.progressElem.innerHTML = "Done"

            } catch (e) {
                console.log("ERR:", e)
                window.appLogger.log(e)
            } finally {
                Array.from(toolsList.querySelectorAll("button")).forEach(button => {
                    button.disabled = false
                })
            }

            prepAudioSpinner.style.display = "none"
            window.tools_state.running = false
            window.refreshRecordsList(window.appState.currentDataset)
        }
    })
})

// Preprocess text
prepTextStart.addEventListener("click", () => {

    // Disable the tool buttons, to prevent parallel execution of other tools/tool instances
    Array.from(toolsList.querySelectorAll("button")).forEach(button => {
        button.disabled = true
    })

    prepTextSpinner.style.display = "inline-block"

    try {
        const DOING_BACKUP = prepTextBackup.checked
        const FILTERING_BLANKS = prepTestFilterBlank.checked
        const FILTERING_LINES_WITH_CHARS = prepTestFilterChars.checked
        const BAD_CHARS = prepTestFilterCharsList.value.split(",")
        const FILTERING_DUPLICATES = prepTestFilterDuplicates.checked

        const newMetadata = []

        if (DOING_BACKUP) {
            fs.copyFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata_backup.csv`)
        }

        const audioFileNamesSoFar = new Set()
        const duplicatesToRemove = []

        const originalLines = fs.readFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`, "utf8").split("\n")
        originalLines.forEach(line => {
            if (!line.length) {
                return
            }

            if (FILTERING_BLANKS && line.includes("||")) {
                return
            }

            const parts = line.split("|")
            const audioFileName = parts[0]
            const text = parts[1]

            if (FILTERING_DUPLICATES) {
                if (audioFileNamesSoFar.has(audioFileName)) {
                    duplicatesToRemove.push(audioFileName)
                }
                audioFileNamesSoFar.add(audioFileName)
            }

            if (FILTERING_LINES_WITH_CHARS) {
                let lineIsOk = true
                BAD_CHARS.forEach(badChar => {
                    if (text.includes(badChar)) {
                        lineIsOk = false
                    }
                })
                if (lineIsOk) {
                    newMetadata.push([audioFileName, text])
                }
            } else {
                newMetadata.push([audioFileName, text])
            }
        })

        let filteredFinalLines = []

        if (FILTERING_DUPLICATES) {
            newMetadata.forEach(([audioFileName, text]) => {
                if (!duplicatesToRemove.includes(audioFileName)) {
                    filteredFinalLines.push([audioFileName, text])
                }
            })
        } else {
            filteredFinalLines = newMetadata
        }

        filteredFinalLines = filteredFinalLines.map(line => `${line[0]}|${line[1]}|${line[1]}`).join("\n")

        fs.unlinkSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`)
        fs.writeFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`, filteredFinalLines, "utf8")

    } catch (e) {
        console.log("ERR:", e)
        window.appLogger.log(e)
    } finally {
        Array.from(toolsList.querySelectorAll("button")).forEach(button => {
            button.disabled = false
        })
    }

    prepTextSpinner.style.display = "none"
    window.tools_state.running = false
    window.refreshRecordsList(window.appState.currentDataset)
})


window.wer_cache = {}

checkTextQualityBtn.addEventListener("click", () => {
    if (window.appState.currentDataset!=undefined) {
        confirmModal(`Are you sure you'd like to kick off the text quality evaluation? This includes the auto-transcription as a step.`).then(confirmation => {
            if (confirmation) {
                setTimeout(() => {
                    createModal("spinner", "Auto-transcribing...<br>This may take a few minutes if there are hundreds of lines.<br>Audio files must be mono 22050Hz<br><br>This window will close if there is an error.")
                    window.appState.skipRefreshing = true


                    const inputDirectory =  `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`

                    window.tools_state.taskId = "transcribe"
                    window.tools_state.inputDirectory = inputDirectory
                    window.tools_state.outputDirectory = `${window.path}/python/transcribe/output`
                    window.tools_state.inputFileType = "folder"

                    window.tools_state.spinnerElem = toolsSpinner
                    window.tools_state.progressElem = toolProgressInfo
                    window.tools_state.infoElem = toolsInfo
                    window.tools_state.currentFileElem = toolsCurrentFile

                    window.tools_state.toolSettings["transcribe"] = window.tools_state.toolSettings["transcribe"] || {}
                    window.tools_state.toolSettings["transcribe"].ignore_existing_transcript = true

                    window.tools_state.post_callback = () => {
                        window.appState.skipRefreshing = true

                        // Copy the two transcript files into the tool directory
                        fs.copyFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`, `${window.path}/python/wer_evaluation/input/metadata.csv`)
                        fs.copyFileSync(`${window.path}/python/transcribe/output/metadata.csv`, `${window.path}/python/wer_evaluation/asr_reference/metadata.csv`)

                        window.tools_state.taskId = "wer_evaluation"
                        window.tools_state.inputDirectory = `${window.path}/python/wer_evaluation/input`
                        window.tools_state.inputDirectory2 = `${window.path}/python/wer_evaluation/asr_reference`
                        window.tools_state.outputDirectory = `${window.path}/python/wer_evaluation/output`


                        window.tools_state.post_callback = () => {

                            window.tools_state.toolSettings["transcribe"].ignore_existing_transcript = false

                            const wer_eval_data = fs.readFileSync(`${window.tools_state.outputDirectory}/wer_eval_results.txt`, "utf8")
                            const wer_key = {}
                            wer_eval_data.split("\n").forEach(row => {
                                const fname = row.split("|")[0].toLowerCase()
                                const score = row.split("|")[1]
                                 wer_key[fname] = score
                            })

                            window.wer_cache[window.appState.currentDataset] = {}

                            window.datasets[window.appState.currentDataset].metadata.forEach((sampleItems, si) => {

                                const fileName = sampleItems[0].fileName.toLowerCase()

                                if (wer_key[fileName]==undefined) {
                                    return
                                }


                                const wer_elem = sampleItems[1].children[4]
                                const score = parseFloat(wer_key[fileName].trim())/2

                                window.wer_cache[window.appState.currentDataset][si] = score

                                window.datasets[window.appState.currentDataset].metadata.wer = score

                                // const r_col = Math.min(score, 1)
                                // const g_col = 1 - r_col
                                // wer_elem.style.background = `rgba(${r_col*255},${g_col*255},50, 0.7)`
                            })

                            window.refreshRecordsList(window.appState.currentDataset)
                            window.appState.skipRefreshing = false
                            closeModal()
                        }

                        toolsRunTool.click()
                    }

                    toolsRunTool.click()

                }, 300)
            }
        })
    }
})



// Clean the data
cleanTextAudioRunBtn.addEventListener("click", () => {

    window.tools_state.running = true

    const metadataAudioFiles = new Set()

    // Remove extra text lines
    const textLines = fs.readFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`, "utf8").split("\n")
    const newMetadata = []
    textLines.forEach(line => {
        const audioFileName = line.split("|")[0]
        metadataAudioFiles.add(audioFileName)

        if (fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${audioFileName}`)) {
            newMetadata.push(line)
        }
    })
    if (textLines.length != newMetadata.length) {
        fs.unlinkSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`)
        fs.writeFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`, newMetadata.join("\n"), "utf8")
    }


    // Remove extra audio files
    const audioFiles = fs.readdirSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/`)
    audioFiles.forEach(fileName => {
        if (!metadataAudioFiles.has(fileName)) {
            fs.unlink(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${fileName}`)
        }
    })

    window.tools_state.running = false
    window.refreshRecordsList(window.appState.currentDataset)
})


