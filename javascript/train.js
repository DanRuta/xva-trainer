"use strict"

const smi = require('node-nvidia-smi')

// https://www.chartjs.org/docs/latest/





setTimeout(() => {

    const startingData = []
    for (let i=0; i<60; i++) {
        startingData.push(100-i)
    }

    window.loss_chart_object = window.initChart("loss_plot", "Loss", "255,112,67", {startingData})
    window.loss_delta_chart_object = window.initChart("loss_delta_plot", "Loss Delta", "255,112,67", {startingData})


    let i = 1

    setInterval(() => {
        window.loss_chart_object.data.datasets[0].data.push(100/i)
        window.loss_chart_object.data.datasets[0].data.splice(0,1)
        window.loss_delta_chart_object.data.datasets[0].data.push(100/i)
        window.loss_delta_chart_object.data.datasets[0].data.splice(0,1)
        i++

        window.loss_chart_object.update()
        window.loss_delta_chart_object.update()

    }, 1000)

}, 2000)




const lorem_cmd = `
Epoch: 4680 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4680_49200.pt | frames/s: 48088 | Loss: 8.16642 | Delta: 0.00271 | Target: 0.0002<br>
Epoch: 4685 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4685_49450.pt | frames/s: 46615 | Loss: 8.06057 | Delta: 0.00261 | Target: 0.0002<br>
Epoch: 4690 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4690_49700.pt | frames/s: 48447 | Loss: 7.96336 | Delta: 0.00251 | Target: 0.0002<br>
Epoch: 4695 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4695_49950.pt | frames/s: 40350 | Loss: 7.86858 | Delta: 0.00237 | Target: 0.0002<br>
Epoch: 4700 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4700_50200.pt | frames/s: 47828 | Loss: 7.77781 | Delta: 0.00232 | Target: 0.0002<br>
Epoch: 4705 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4705_50450.pt | frames/s: 46347 | Loss: 7.69099 | Delta: 0.00219 | Target: 0.0002<br>
Epoch: 4710 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4710_50700.pt | frames/s: 47361 | Loss: 7.6075 | Delta: 0.00217 | Target: 0.0002<br>
Epoch: 4715 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4715_50950.pt | frames/s: 47573 | Loss: 7.52751 | Delta: 0.0021 | Target: 0.0002<br>
Epoch: 4720 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4720_51200.pt | frames/s: 47223 | Loss: 7.4513 | Delta: 0.00203 | Target: 0.0002<br>
Epoch: 4725 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4725_51450.pt | frames/s: 49138 | Loss: 7.37719 | Delta: 0.00199 | Target: 0.0002<br>
Epoch: 4730 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4730_51700.pt | frames/s: 49215 | Loss: 7.30604 | Delta: 0.00189 | Target: 0.0002<br>
Epoch: 4735 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4735_51950.pt | frames/s: 49039 | Loss: 7.23754 | Delta: 0.00187 | Target: 0.0002<br>
Epoch: 4740 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4740_52200.pt | frames/s: 48227 | Loss: 7.17171 | Delta: 0.00183 | Target: 0.0002<br>
Epoch: 4745 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4745_52450.pt | frames/s: 47248 | Loss: 7.10808 | Delta: 0.00176 | Target: 0.0002<br>
Epoch: 4750 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4750_52700.pt | frames/s: 48713 | Loss: 7.04659 | Delta: 0.00172 | Target: 0.0002<br>
Epoch: 4755 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4755_52950.pt | frames/s: 49460 | Loss: 6.98719 | Delta: 0.00168 | Target: 0.0002<br>
Epoch: 4760 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4760_53200.pt | frames/s: 48304 | Loss: 6.9297 | Delta: 0.00165 | Target: 0.0002<br>
Epoch: 4765 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4765_53450.pt | frames/s: 47651 | Loss: 6.87388 | Delta: 0.0016 | Target: 0.0002<br>
Epoch: 4770 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4770_53700.pt | frames/s: 48525 | Loss: 6.81979 | Delta: 0.00157 | Target: 0.0002<br>
Epoch: 4775 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4775_53950.pt | frames/s: 48632 | Loss: 6.76753 | Delta: 0.00152 | Target: 0.0002<br>
Epoch: 4780 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4780_54200.pt | frames/s: 49094 | Loss: 6.71674 | Delta: 0.00149 | Target: 0.0002<br>
Epoch: 4785 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4785_54450.pt | frames/s: 49187 | Loss: 6.66757 | Delta: 0.00146 | Target: 0.0002<br>
Epoch: 4790 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4790_54700.pt | frames/s: 48907 | Loss: 6.61969 | Delta: 0.00143 | Target: 0.0002<br>
Epoch: 4795 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4795_54950.pt | frames/s: 49485 | Loss: 6.5732 | Delta: 0.00141 | Target: 0.0002<br>
Epoch: 4800 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4800_55200.pt | frames/s: 48798 | Loss: 6.52791 | Delta: 0.00137 | Target: 0.0002<br>
Epoch: 4805 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4805_55450.pt | frames/s: 48844 | Loss: 6.48379 | Delta: 0.00135 | Target: 0.0002<br>
Epoch: 4810 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4810_55700.pt | frames/s: 49854 | Loss: 6.44102 | Delta: 0.00131 | Target: 0.0002<br>
Epoch: 4815 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4815_55950.pt | frames/s: 48678 | Loss: 6.39929 | Delta: 0.0013 | Target: 0.0002<br>
Epoch: 4820 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4820_56200.pt | frames/s: 49101 | Loss: 6.35844 | Delta: 0.00127 | Target: 0.0002<br>
Epoch: 4825 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4825_56450.pt | frames/s: 49224 | Loss: 6.31865 | Delta: 0.00125 | Target: 0.0002<br>
Epoch: 4830 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4830_56700.pt | frames/s: 49353 | Loss: 6.27988 | Delta: 0.00123 | Target: 0.0002<br>
Epoch: 4835 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4835_56950.pt | frames/s: 48765 | Loss: 6.24209 | Delta: 0.0012 | Target: 0.0002<br>
Epoch: 4840 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4840_57200.pt | frames/s: 49456 | Loss: 6.20515 | Delta: 0.00117 | Target: 0.0002<br>
Epoch: 4845 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4845_57450.pt | frames/s: 49151 | Loss: 6.16915 | Delta: 0.00115 | Target: 0.0002<br>
Epoch: 4850 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4850_57700.pt | frames/s: 48477 | Loss: 6.13385 | Delta: 0.00114 | Target: 0.0002<br>
Epoch: 4855 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4855_57950.pt | frames/s: 49096 | Loss: 6.09935 | Delta: 0.00112 | Target: 0.0002<br>
Epoch: 4860 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4860_58200.pt | frames/s: 49317 | Loss: 6.06557 | Delta: 0.0011 | Target: 0.0002<br>
Epoch: 4865 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4865_58450.pt | frames/s: 49251 | Loss: 6.03248 | Delta: 0.00108 | Target: 0.0002<br>
Epoch: 4870 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4870_58700.pt | frames/s: 48697 | Loss: 6.00016 | Delta: 0.00107 | Target: 0.0002<br>
Epoch: 4875 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4875_58950.pt | frames/s: 48997 | Loss: 5.96854 | Delta: 0.00104 | Target: 0.0002<br>
Epoch: 4880 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4880_59200.pt | frames/s: 49387 | Loss: 5.93751 | Delta: 0.00103 | Target: 0.0002<br>
Epoch: 4885 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4885_59450.pt | frames/s: 48884 | Loss: 5.90714 | Delta: 0.00102 | Target: 0.0002<br>
Epoch: 4890 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4890_59700.pt | frames/s: 49350 | Loss: 5.87733 | Delta: 0.001 | Target: 0.0002<br>
Epoch: 4895 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4895_59950.pt | frames/s: 48251 | Loss: 5.84813 | Delta: 0.00099 | Target: 0.0002<br>
Epoch: 4900 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4900_60200.pt | frames/s: 49269 | Loss: 5.81953 | Delta: 0.00097 | Target: 0.0002<br>
Epoch: 4905 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4905_60450.pt | frames/s: 48162 | Loss: 5.79145 | Delta: 0.00096 | Target: 0.0002<br>
Epoch: 4910 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4910_60700.pt | frames/s: 48899 | Loss: 5.76394 | Delta: 0.00094 | Target: 0.0002<br>
Epoch: 4915 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4915_60950.pt | frames/s: 49046 | Loss: 5.73695 | Delta: 0.00093 | Target: 0.0002<br>
Epoch: 4920 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4920_61200.pt | frames/s: 49141 | Loss: 5.71043 | Delta: 0.00092 | Target: 0.0002<br>
Epoch: 4925 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4925_61450.pt | frames/s: 48689 | Loss: 5.6844 | Delta: 0.00091 | Target: 0.0002<br>
Epoch: 4930 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4930_61700.pt | frames/s: 49149 | Loss: 5.65886 | Delta: 0.00089 | Target: 0.0002<br>
Epoch: 4935 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4935_61950.pt | frames/s: 48906 | Loss: 5.63374 | Delta: 0.00088 | Target: 0.0002<br>
Epoch: 4940 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4940_62200.pt | frames/s: 48060 | Loss: 5.60904 | Delta: 0.00087 | Target: 0.0002<br>
Epoch: 4945 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4945_62450.pt | frames/s: 49142 | Loss: 5.5848 | Delta: 0.00086 | Target: 0.0002<br>
Epoch: 4950 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4950_62700.pt | frames/s: 48717 | Loss: 5.56097 | Delta: 0.00085 | Target: 0.0002<br>
Epoch: 4955 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4955_62950.pt | frames/s: 47623 | Loss: 5.53751 | Delta: 0.00084 | Target: 0.0002<br>
Epoch: 4960 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4960_63200.pt | frames/s: 48934 | Loss: 5.51449 | Delta: 0.00082 | Target: 0.0002<br>
Epoch: 4965 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4965_63450.pt | frames/s: 48832 | Loss: 5.49184 | Delta: 0.00082 | Target: 0.0002<br>
Epoch: 4970 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4970_63700.pt | frames/s: 47823 | Loss: 5.46957 | Delta: 0.00081 | Target: 0.0002<br>
Epoch: 4975 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4975_63950.pt | frames/s: 47176 | Loss: 5.44764 | Delta: 0.0008 | Target: 0.0002<br>
Epoch: 4980 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4980_64200.pt | frames/s: 46028 | Loss: 5.42603 | Delta: 0.00079 | Target: 0.0002<br>
Epoch: 4985 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4985_64450.pt | frames/s: 43086 | Loss: 5.40481 | Delta: 0.00078 | Target: 0.0002<br>
Epoch: 4990 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4990_64700.pt | frames/s: 48453 | Loss: 5.38394 | Delta: 0.00077 | Target: 0.0002<br>
Epoch: 4995 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~4995_64950.pt | frames/s: 49147 | Loss: 5.36337 | Delta: 0.00076 | Target: 0.0002<br>
Epoch: 5000 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5000_65200.pt | frames/s: 48810 | Loss: 5.3431 | Delta: 0.00075 | Target: 0.0002<br>
Epoch: 5005 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5005_65450.pt | frames/s: 49006 | Loss: 5.32315 | Delta: 0.00074 | Target: 0.0002<br>
Epoch: 5010 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5010_65700.pt | frames/s: 48634 | Loss: 5.30348 | Delta: 0.00073 | Target: 0.0002<br>
Epoch: 5015 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5015_65950.pt | frames/s: 47593 | Loss: 5.28411 | Delta: 0.00072 | Target: 0.0002<br>
Epoch: 5020 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5020_66200.pt | frames/s: 48565 | Loss: 5.26501 | Delta: 0.00072 | Target: 0.0002<br>
Epoch: 5025 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5025_66450.pt | frames/s: 43221 | Loss: 5.24621 | Delta: 0.00071 | Target: 0.0002<br>
Epoch: 5030 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5030_66700.pt | frames/s: 43437 | Loss: 5.22771 | Delta: 0.0007 | Target: 0.0002<br>
Epoch: 5035 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5035_66950.pt | frames/s: 35246 | Loss: 5.20944 | Delta: 0.00069 | Target: 0.0002<br>
Epoch: 5040 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5040_67200.pt | frames/s: 35590 | Loss: 5.19142 | Delta: 0.00069 | Target: 0.0002<br>
Epoch: 5045 | Stage: 4 | H:/xVA/FP_OUTPUT/voice_name~5045_67450.pt | frames/s: 33255 | Loss: 5.17369 | Delta: 0.00068 | Target: 0.0002<br>
Stage: 4 | Epoch: 5048 | iter: 32/50 -> 67581 | loss: 3.8518 | frames/s 37990 | Avg loss delta: 0.00068 | Target: 0.0002`
cmdPanel.innerHTML = lorem_cmd+"<br><br><br><br><br>"


// window.updateSystemGraphs = () => {

//     const totalRam = window.totalmem()
//     const ramUsage = window.totalmem() - window.freemem()
//     const percentRAM = ramUsage/totalRam*100


//     window.cpuUsage(cpuLoad => {
//         smi((err, data) => {
//             const totalVRAM = parseInt(data.nvidia_smi_log.gpu.fb_memory_usage.total.split(" ")[0])
//             const usedVRAM = parseInt(data.nvidia_smi_log.gpu.fb_memory_usage.used.split(" ")[0])
//             const percent = usedVRAM/totalVRAM*100

//             let vramUsage = `${(usedVRAM/1000).toFixed(1)}/${(totalVRAM/1000).toFixed(1)} GB (${percent.toFixed(2)}%)`

//             const computeLoad = parseFloat(data.nvidia_smi_log.gpu.utilization.gpu_util.split(" ")[0])


//             console.log(`RAM: ${parseInt(ramUsage)}GB/${parseInt(totalRam)}GB (${percentRAM.toFixed(2)}%) | CPU: ${(cpuLoad*100).toFixed(2)}% | GPU: ${computeLoad}% | VRAM: ${vramUsage}`)
//         })
//     })
// }