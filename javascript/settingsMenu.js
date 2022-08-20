"use strict"

const saveUserSettings = () => localStorage.setItem("userSettings", JSON.stringify(window.userSettings))

const deleteFolderRecursive = function (directoryPath) {
    if (fs.existsSync(directoryPath)) {
        fs.readdirSync(directoryPath).forEach((file, index) => {
          const curPath = `${directoryPath}/${file}`;
          if (fs.lstatSync(curPath).isDirectory()) {
           // recurse
            deleteFolderRecursive(curPath);
          } else {
            // delete file
            fs.unlinkSync(curPath);
          }
        });
        fs.rmdirSync(directoryPath);
      }
    };

// Load user settings
window.userSettings = localStorage.getItem("userSettings") ||
    {
        customWindowSize:`${window.innerHeight},${window.innerWidth}`,
    }
if ((typeof window.userSettings)=="string") {
    window.userSettings = JSON.parse(window.userSettings)
}

if (!Object.keys(window.userSettings).includes("installation")) { // For backwards compatibility
    window.userSettings.installation = "gpu"
}
if (!Object.keys(window.userSettings).includes("paginationSize")) { // For backwards compatibility
    window.userSettings.paginationSize = 100
}
if (!Object.keys(window.userSettings).includes("outpathwidth")) { // For backwards compatibility
    window.userSettings.outpathwidth = 250
}
if (!Object.keys(window.userSettings).includes("micVolume")) { // For backwards compatibility
    window.userSettings.micVolume = 250
}
if (!Object.keys(window.userSettings).includes("removeNoise")) { // For backwards compatibility
    window.userSettings.removeNoise = false
}
if (!Object.keys(window.userSettings).includes("noiseRemStrength")) { // For backwards compatibility
    window.userSettings.noiseRemStrength = 0.25
}
if (!Object.keys(window.userSettings).includes("cudaDevices")) { // For backwards compatibility
    window.userSettings.cudaDevices = 0
}
if (!Object.keys(window.userSettings).includes("datasetsPath")) { // For backwards compatibility
    window.userSettings.datasetsPath = `${__dirname.replace(/\\/g,"/")}/datasets/`.replace(/\/\//g, "/").replace("resources/app/resources/app", "resources/app").replace("/javascript", "")

}

const updateUIWithSettings = () => {
    // useGPUCbx.checked = window.userSettings.useGPU
    const [height, width] = window.userSettings.customWindowSize.split(",").map(v => parseInt(v))
    ipcRenderer.send("resize", {height, width})
    setting_paginationSize.value = window.userSettings.paginationSize
    setting_outpathwidth.value = window.userSettings.outpathwidth
    setting_micVolume.value = window.userSettings.micVolume
    setting_micVolume_display.innerHTML = `${parseInt(setting_micVolume.value)}%`
    setting_doNoiseRemoval.checked = window.userSettings.removeNoise
    setting_noiseRemStrength.value = window.userSettings.noiseRemStrength
    setting_cudaDevices.value = window.userSettings.cudaDevices

    setting_datasets_input.value = window.userSettings.datasetsPath
}
updateUIWithSettings()
saveUserSettings()

// Add the SVG code this way, because otherwise the index.html file will be spammed with way too much svg code
Array.from(window.document.querySelectorAll(".svgButton")).forEach(svgButton => {
    svgButton.innerHTML = `<svg class="openFolderSVG" width="400" height="350" viewBox="0, 0, 400,350"><g id="svgg" ><path id="path0"  d="M39.960 53.003 C 36.442 53.516,35.992 53.635,30.800 55.422 C 15.784 60.591,3.913 74.835,0.636 91.617 C -0.372 96.776,-0.146 305.978,0.872 310.000 C 5.229 327.228,16.605 339.940,32.351 345.172 C 40.175 347.773,32.175 347.630,163.000 347.498 L 281.800 347.378 285.600 346.495 C 304.672 342.065,321.061 332.312,330.218 319.944 C 330.648 319.362,332.162 317.472,333.581 315.744 C 335.001 314.015,336.299 312.420,336.467 312.200 C 336.634 311.980,337.543 310.879,338.486 309.753 C 340.489 307.360,342.127 305.341,343.800 303.201 C 344.460 302.356,346.890 299.375,349.200 296.575 C 351.510 293.776,353.940 290.806,354.600 289.975 C 355.260 289.144,356.561 287.505,357.492 286.332 C 358.422 285.160,359.952 283.267,360.892 282.126 C 362.517 280.153,371.130 269.561,375.632 264.000 C 376.789 262.570,380.427 258.097,383.715 254.059 C 393.790 241.689,396.099 237.993,398.474 230.445 C 403.970 212.972,394.149 194.684,376.212 188.991 C 369.142 186.747,368.803 186.724,344.733 186.779 C 330.095 186.812,322.380 186.691,322.216 186.425 C 322.078 186.203,321.971 178.951,321.977 170.310 C 321.995 146.255,321.401 141.613,317.200 133.000 C 314.009 126.457,307.690 118.680,303.142 115.694 C 302.560 115.313,301.300 114.438,300.342 113.752 C 295.986 110.631,288.986 107.881,282.402 106.704 C 280.540 106.371,262.906 106.176,220.400 106.019 L 161.000 105.800 160.763 98.800 C 159.961 75.055,143.463 56.235,120.600 52.984 C 115.148 52.208,45.292 52.225,39.960 53.003 M120.348 80.330 C 130.472 83.988,133.993 90.369,133.998 105.071 C 134.003 120.968,137.334 127.726,147.110 131.675 L 149.400 132.600 213.800 132.807 C 272.726 132.996,278.392 133.071,280.453 133.690 C 286.872 135.615,292.306 141.010,294.261 147.400 C 294.928 149.578,294.996 151.483,294.998 168.000 L 295.000 186.200 292.800 186.449 C 291.590 186.585,254.330 186.725,210.000 186.759 C 163.866 186.795,128.374 186.977,127.000 187.186 C 115.800 188.887,104.936 192.929,96.705 198.458 C 95.442 199.306,94.302 200.000,94.171 200.000 C 93.815 200.000,89.287 203.526,87.000 205.583 C 84.269 208.039,80.083 212.649,76.488 217.159 C 72.902 221.657,72.598 222.031,70.800 224.169 C 70.030 225.084,68.770 226.620,68.000 227.582 C 67.230 228.544,66.054 229.977,65.387 230.766 C 64.720 231.554,62.727 234.000,60.957 236.200 C 59.188 238.400,56.346 241.910,54.642 244.000 C 52.938 246.090,50.163 249.510,48.476 251.600 C 44.000 257.146,36.689 266.126,36.212 266.665 C 35.985 266.921,34.900 268.252,33.800 269.623 C 32.700 270.994,30.947 273.125,29.904 274.358 C 28.861 275.591,28.006 276.735,28.004 276.900 C 28.002 277.065,27.728 277.200,27.395 277.200 C 26.428 277.200,26.700 96.271,27.670 93.553 C 30.020 86.972,35.122 81.823,40.800 80.300 C 44.238 79.378,47.793 79.296,81.800 79.351 L 117.800 79.410 120.348 80.330 M369.400 214.800 C 374.239 217.220,374.273 222.468,369.489 228.785 C 367.767 231.059,364.761 234.844,364.394 235.200 C 364.281 235.310,362.373 237.650,360.154 240.400 C 357.936 243.150,354.248 247.707,351.960 250.526 C 347.732 255.736,346.053 257.821,343.202 261.400 C 341.505 263.530,340.849 264.336,334.600 271.965 C 332.400 274.651,330.204 277.390,329.720 278.053 C 329.236 278.716,328.246 279.945,327.520 280.785 C 326.794 281.624,325.300 283.429,324.200 284.794 C 323.100 286.160,321.726 287.845,321.147 288.538 C 320.568 289.232,318.858 291.345,317.347 293.233 C 308.372 304.449,306.512 306.609,303.703 309.081 C 299.300 312.956,290.855 317.633,286.000 318.886 C 277.958 320.960,287.753 320.819,159.845 320.699 C 33.557 320.581,42.330 320.726,38.536 318.694 C 34.021 316.276,35.345 310.414,42.386 301.647 C 44.044 299.583,45.940 297.210,46.600 296.374 C 47.260 295.538,48.340 294.169,49.000 293.332 C 49.660 292.495,51.550 290.171,53.200 288.167 C 54.850 286.164,57.100 283.395,58.200 282.015 C 59.300 280.635,60.920 278.632,61.800 277.564 C 62.680 276.496,64.210 274.617,65.200 273.389 C 66.190 272.162,67.188 270.942,67.418 270.678 C 67.649 270.415,71.591 265.520,76.179 259.800 C 80.767 254.080,84.634 249.310,84.773 249.200 C 84.913 249.090,87.117 246.390,89.673 243.200 C 92.228 240.010,95.621 235.780,97.213 233.800 C 106.328 222.459,116.884 215.713,128.200 213.998 C 129.300 213.832,183.570 213.719,248.800 213.748 L 367.400 213.800 369.400 214.800 " stroke="none" fill="#fbfbfb" fill-rule="evenodd"></path><path id="path1" fill-opacity="0" d="M0.000 46.800 C 0.000 72.540,0.072 93.600,0.159 93.600 C 0.246 93.600,0.516 92.460,0.759 91.066 C 3.484 75.417,16.060 60.496,30.800 55.422 C 35.953 53.648,36.338 53.550,40.317 52.981 C 46.066 52.159,114.817 52.161,120.600 52.984 C 143.463 56.235,159.961 75.055,160.763 98.800 L 161.000 105.800 220.400 106.019 C 262.906 106.176,280.540 106.371,282.402 106.704 C 288.986 107.881,295.986 110.631,300.342 113.752 C 301.300 114.438,302.560 115.313,303.142 115.694 C 307.690 118.680,314.009 126.457,317.200 133.000 C 321.401 141.613,321.995 146.255,321.977 170.310 C 321.971 178.951,322.078 186.203,322.216 186.425 C 322.380 186.691,330.095 186.812,344.733 186.779 C 368.803 186.724,369.142 186.747,376.212 188.991 C 381.954 190.814,388.211 194.832,391.662 198.914 C 395.916 203.945,397.373 206.765,399.354 213.800 C 399.842 215.533,399.922 201.399,399.958 107.900 L 400.000 0.000 200.000 0.000 L 0.000 0.000 0.000 46.800 M44.000 79.609 C 35.903 81.030,30.492 85.651,27.670 93.553 C 26.700 96.271,26.428 277.200,27.395 277.200 C 27.728 277.200,28.002 277.065,28.004 276.900 C 28.006 276.735,28.861 275.591,29.904 274.358 C 30.947 273.125,32.700 270.994,33.800 269.623 C 34.900 268.252,35.985 266.921,36.212 266.665 C 36.689 266.126,44.000 257.146,48.476 251.600 C 50.163 249.510,52.938 246.090,54.642 244.000 C 56.346 241.910,59.188 238.400,60.957 236.200 C 62.727 234.000,64.720 231.554,65.387 230.766 C 66.054 229.977,67.230 228.544,68.000 227.582 C 68.770 226.620,70.030 225.084,70.800 224.169 C 72.598 222.031,72.902 221.657,76.488 217.159 C 80.083 212.649,84.269 208.039,87.000 205.583 C 89.287 203.526,93.815 200.000,94.171 200.000 C 94.302 200.000,95.442 199.306,96.705 198.458 C 104.936 192.929,115.800 188.887,127.000 187.186 C 128.374 186.977,163.866 186.795,210.000 186.759 C 254.330 186.725,291.590 186.585,292.800 186.449 L 295.000 186.200 294.998 168.000 C 294.996 151.483,294.928 149.578,294.261 147.400 C 292.306 141.010,286.872 135.615,280.453 133.690 C 278.392 133.071,272.726 132.996,213.800 132.807 L 149.400 132.600 147.110 131.675 C 137.334 127.726,134.003 120.968,133.998 105.071 C 133.993 90.369,130.472 83.988,120.348 80.330 L 117.800 79.410 81.800 79.351 C 62.000 79.319,44.990 79.435,44.000 79.609 M128.200 213.998 C 116.884 215.713,106.328 222.459,97.213 233.800 C 95.621 235.780,92.228 240.010,89.673 243.200 C 87.117 246.390,84.913 249.090,84.773 249.200 C 84.634 249.310,80.767 254.080,76.179 259.800 C 71.591 265.520,67.649 270.415,67.418 270.678 C 67.188 270.942,66.190 272.162,65.200 273.389 C 64.210 274.617,62.680 276.496,61.800 277.564 C 60.920 278.632,59.300 280.635,58.200 282.015 C 57.100 283.395,54.850 286.164,53.200 288.167 C 51.550 290.171,49.660 292.495,49.000 293.332 C 48.340 294.169,47.260 295.538,46.600 296.374 C 45.940 297.210,44.044 299.583,42.386 301.647 C 35.345 310.414,34.021 316.276,38.536 318.694 C 42.330 320.726,33.557 320.581,159.845 320.699 C 287.753 320.819,277.958 320.960,286.000 318.886 C 290.855 317.633,299.300 312.956,303.703 309.081 C 306.512 306.609,308.372 304.449,317.347 293.233 C 318.858 291.345,320.568 289.232,321.147 288.538 C 321.726 287.845,323.100 286.160,324.200 284.794 C 325.300 283.429,326.794 281.624,327.520 280.785 C 328.246 279.945,329.236 278.716,329.720 278.053 C 330.204 277.390,332.400 274.651,334.600 271.965 C 340.849 264.336,341.505 263.530,343.202 261.400 C 346.053 257.821,347.732 255.736,351.960 250.526 C 354.248 247.707,357.936 243.150,360.154 240.400 C 362.373 237.650,364.281 235.310,364.394 235.200 C 364.761 234.844,367.767 231.059,369.489 228.785 C 374.273 222.468,374.239 217.220,369.400 214.800 L 367.400 213.800 248.800 213.748 C 183.570 213.719,129.300 213.832,128.200 213.998 M399.600 225.751 C 399.600 231.796,394.623 240.665,383.715 254.059 C 380.427 258.097,376.789 262.570,375.632 264.000 C 371.130 269.561,362.517 280.153,360.892 282.126 C 359.952 283.267,358.422 285.160,357.492 286.332 C 356.561 287.505,355.260 289.144,354.600 289.975 C 353.940 290.806,351.510 293.776,349.200 296.575 C 346.890 299.375,344.460 302.356,343.800 303.201 C 342.127 305.341,340.489 307.360,338.486 309.753 C 337.543 310.879,336.634 311.980,336.467 312.200 C 336.299 312.420,335.001 314.015,333.581 315.744 C 332.162 317.472,330.648 319.362,330.218 319.944 C 321.061 332.312,304.672 342.065,285.600 346.495 L 281.800 347.378 163.000 347.498 C 32.175 347.630,40.175 347.773,32.351 345.172 C 16.471 339.895,3.810 325.502,0.820 309.326 C 0.591 308.085,0.312 306.979,0.202 306.868 C 0.091 306.757,-0.000 327.667,-0.000 353.333 L 0.000 400.000 200.000 400.000 L 400.000 400.000 400.000 312.400 C 400.000 264.220,399.910 224.800,399.800 224.800 C 399.690 224.800,399.600 225.228,399.600 225.751 " stroke="none" fill="#050505" fill-rule="evenodd"></path></g></svg>`
})

const initMenuSetting = (elem, setting, type, callback=undefined, valFn=undefined) => {

    valFn = valFn ? valFn : x=>x

    if (type=="checkbox") {
        elem.addEventListener("click", () => {
            if (setting.includes(".")) {
                window.userSettings[setting.split(".")[0]][setting.split(".")[1]] = valFn(elem.checked)
            } else {
                window.userSettings[setting] = valFn(elem.checked)
            }
            saveUserSettings()
            if (callback) callback()
        })
    } else {
        elem.addEventListener("change", () => {
            if (setting.includes(".")) {
                window.userSettings[setting.split(".")[0]][setting.split(".")[1]] = valFn(elem.value)
            } else {
                window.userSettings[setting] = valFn(elem.value)
            }
            saveUserSettings()
            if (callback) callback()
        })
    }
}

const setPromptTheme = () => {
    if (window.userSettings.darkPrompt) {
        dialogueInput.style.backgroundColor = "rgba(25,25,25,0.9)"
        dialogueInput.style.color = "white"
    } else {
        dialogueInput.style.backgroundColor = "rgba(255,255,255,0.9)"
        dialogueInput.style.color = "black"
    }
}
const setPromptFontSize = () => {
    dialogueInput.style.fontSize = `${window.userSettings.prompt_fontSize}pt`
}
const updateBackground = () => {
    const background = `linear-gradient(0deg, rgba(128,128,128,${window.userSettings.bg_gradient_opacity}) 0px, rgba(0,0,0,0)), url("assets/${window.currentGame.join("-")}")`
    // Fade the background image transition
    rightBG1.style.background = background
    rightBG2.style.opacity = 0
    setTimeout(() => {
        rightBG2.style.background = rightBG1.style.background
        rightBG2.style.opacity = 1
    }, 1000)
}

initMenuSetting(setting_paginationSize, "paginationSize", "number", undefined, parseInt)
initMenuSetting(setting_outpathwidth, "outpathwidth", "number", () => {
    outPathCSSHack.innerHTML = `
    #batchRecordsHeader > div:nth-child(3), #batchRecordsContainer > div > div:nth-child(3) {
        min-width: ${parseInt(setting_outpathwidth.value)}px;
        max-width: ${parseInt(setting_outpathwidth.value)}px;
    }
    `
}, parseInt)
initMenuSetting(setting_micVolume, "micVolume", "number", ()=>{
    setting_micVolume_display.innerHTML = `${parseInt(setting_micVolume.value)}%`
    window.initMic()
}, parseInt)
initMenuSetting(setting_doNoiseRemoval, "removeNoise", "checkbox")
initMenuSetting(setting_noiseRemStrength, "noiseRemStrength", "number", parseFloat)
initMenuSetting(setting_cudaDevices, "cudaDevices", "number", parseFloat)
initMenuSetting(setting_datasets_input, "datasetsPath", "text")


reset_settings_btn.addEventListener("click", () => {
    window.confirmModal(window.i18n.SURE_RESET_SETTINGS).then(confirmation => {
        if (confirmation) {
            // window.userSettings.audio.format = "wav"

            updateUIWithSettings()
            saveUserSettings()
        }
    })
})

initFilePickerButton(datasetsPathButton, setting_datasets_input, "datasetsPath", ["openDirectory"], undefined, undefined, ()=>window.refreshDatasets())
window.initFilePickerButton(trainingAddConfigDatasetPathInputBtn, trainingAddConfigDatasetPathInput, undefined, ["openDirectory"], undefined, window.userSettings.datasetsPath)
window.initFilePickerButton(trainingAddConfigOutputPathInputBtn, trainingAddConfigOutputPathInput, undefined, ["openDirectory"], undefined, window.path)
window.initFilePickerButton(modelExport_trainningDirBtn, modelExport_trainningDir, undefined, ["openDirectory"], undefined, window.path)
window.initFilePickerButton(modelExport_outputDirBtn, modelExport_outputDir, undefined, ["openDirectory"], undefined, window.path)
window.initFilePickerButton(trainingAddConfigCkptPathInputBtn, trainingAddConfigCkptPathInput, undefined, ["openFile"], [{name: "Pytorch checkpoint", extensions: ["pt"]}], window.path)

trainingAddConfigBatchSizeInput.addEventListener("keyup", () => window.localStorage.setItem("training.batch_size", trainingAddConfigBatchSizeInput.value))
trainingAddConfigBatchSizeInput.addEventListener("change", () => window.localStorage.setItem("training.batch_size", trainingAddConfigBatchSizeInput.value))
trainingAddConfigBatchSizeInput.value = window.localStorage.getItem("training.batch_size")

trainingAddConfigEpochsPerCkptInput.addEventListener("keyup", () => window.localStorage.setItem("training.epochs_per_ckpt", trainingAddConfigEpochsPerCkptInput.value))
trainingAddConfigEpochsPerCkptInput.addEventListener("change", () => window.localStorage.setItem("training.epochs_per_ckpt", trainingAddConfigEpochsPerCkptInput.value))
trainingAddConfigEpochsPerCkptInput.value = window.localStorage.getItem("training.epochs_per_ckpt")

trainingAddConfigUseAmp.addEventListener("keyup", () => window.localStorage.setItem("training.useFP16", trainingAddConfigUseAmp.checked?"1":"0"))
trainingAddConfigUseAmp.addEventListener("change", () => window.localStorage.setItem("training.useFP16", trainingAddConfigUseAmp.checked?"1":"0"))
trainingAddConfigUseAmp.checked = !!parseInt(window.localStorage.getItem("training.useFP16"))

trainingAddConfigWorkersInput.addEventListener("keyup", () => window.localStorage.setItem("training.num_workers", trainingAddConfigWorkersInput.value))
trainingAddConfigWorkersInput.addEventListener("change", () => window.localStorage.setItem("training.num_workers", trainingAddConfigWorkersInput.value))
trainingAddConfigWorkersInput.value = window.localStorage.getItem("training.num_workers")



const currentWindow = require("electron").remote.getCurrentWindow()
currentWindow.on("move", () => {
    const bounds = remote.getCurrentWindow().webContents.getOwnerBrowserWindow().getBounds()
    window.userSettings.customWindowPosition = `${bounds.x},${bounds.y}`
    saveUserSettings()
})
if (window.userSettings.customWindowPosition) {
    ipcRenderer.send('updatePosition', {details: window.userSettings.customWindowPosition.split(",")})
}



// Installation sever handling
// =========================
settings_installation.innerHTML = window.userSettings.installation=="cpu" ? `CPU` : "CPU+GPU"
setting_change_installation.innerHTML = window.userSettings.installation=="cpu" ? `Change to CPU+GPU` : `Change to CPU`
setting_change_installation.addEventListener("click", () => {
    spinnerModal("Changing installation sever...")
    doFetch(`http://localhost:${window.SERVER_PORT}/stopServer`, {
        method: "Post",
        body: JSON.stringify({})
    }).then(r=>r.text()).then(console.log) // The server stopping should mean this never runs
    .catch(() => {

        if (window.userSettings.installation=="cpu") {
            window.userSettings.installation = "gpu"
            settings_installation.innerHTML = `GPU`
            setting_change_installation.innerHTML = `Change to CPU`
        } else {
            doFetch(`http://localhost:${window.SERVER_PORT}/setDevice`, {
                method: "Post",
                body: JSON.stringify({device: "cpu"})
            })

            window.userSettings.installation = "cpu"
            window.userSettings.useGPU = false
            settings_installation.innerHTML = `CPU`
            setting_change_installation.innerHTML = `Change to CPU+GPU`
        }
        saveUserSettings()

        // Start the new server
        if (window.PRODUCTION) {
            window.appLogger.log(window.userSettings.installation)
            window.pythonProcess = spawn(`${path}/cpython_${window.userSettings.installation}/server.exe`, {stdio: "ignore"})
        } else {
            window.pythonProcess = spawn("python", [`${path}/server.py`], {stdio: "ignore"})
        }

        window.serverIsUp = false
        window.doWeirdServerStartupCheck()
    })
})
window.doWeirdServerStartupCheck = () => {
    const check = () => {
        return new Promise(topResolve => {
            if (window.serverIsUp) {
                topResolve()
            } else {
                (new Promise((resolve, reject) => {
                    // Gather the model paths to send to the server
                    const modelsPaths = {}
                    Object.keys(window.userSettings).filter(key => key.includes("modelspath_")).forEach(key => {
                        modelsPaths[key.split("_")[1]] = window.userSettings[key]
                    })

                    doFetch(`http://localhost:${window.SERVER_PORT}/checkReady`, {
                        method: "Post",
                        body: JSON.stringify({
                            device: (window.userSettings.installation=="gpu")?"gpu":"cpu",
                            modelsPaths: JSON.stringify(modelsPaths)
                        })
                    }).then(r => r.text()).then(r => {
                        closeModal([document.querySelector("#activeModal"), modalContainer], []).then(() => {
                            // window.electronBrowserWindow.setProgressBar(-1)
                        })
                        window.serverIsUp = true
                        if (window.userSettings.installation=="cpu") {
                            saveUserSettings()
                        }

                        resolve()
                    }).catch((err) => {
                        reject()
                    })
                })).catch(() => {
                    setTimeout(async () => {
                        await check()
                        topResolve()
                    }, 100)
                })
            }
        })
    }

    check()
}

window.saveUserSettings = saveUserSettings
exports.saveUserSettings = saveUserSettings
exports.deleteFolderRecursive = deleteFolderRecursive
