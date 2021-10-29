"use strict"

const fs = require("fs-extra")
const {getAudioDurationInSeconds} = require("get-audio-duration")

const tools = {
    "Audio formatting": {
        taskId: "formatting",
        description: "Audio must be in mono 22050Hz .wav format for all tools and models. This tool will automatically convert your audio to this.",
        inputDirectory: `${window.path}/python/audio_format/input`,
        outputDirectory: `${window.path}/python/audio_format/output`,
    },
    "AI speaker diarization": {
        taskId: "diarization",
        description: "Speaker diarization can be used to automatically extract slices of speech audio from a really long audio sample, such as audiobooks and movies. The slices of speech can also be automatically attributed to the correct individual speakers, before being output to file.",
        inputDirectory: `${window.path}/python/speaker_diarization/input`,
        outputDirectory: `${window.path}/python/speaker_diarization/output/`,
        toolSettings: {},
        inputFileType: ".wav",
        setupFn: (taskId) => {
            const ckbxDescription = createElem("div", "Merge output into one folder (Only use when you are sure all files have only one voice)")
            const ckbx = createElem("input", {type: "checkbox"})
            ckbx.style.height = "20px"
            ckbx.style.width = "20px"

            window.tools_state.toolSettings["diarization"] = window.tools_state.toolSettings["diarization"] || {}
            window.tools_state.toolSettings["diarization"].mergeSingleOutputFolder = false

            ckbx.addEventListener("click", () => {
                window.tools_state.toolSettings["diarization"].mergeSingleOutputFolder = ckbx.checked
            })

            const container = createElem("div", ckbx, ckbxDescription)
            container.style.display = "flex"
            container.style.justifyContent = "center"
            container.style.alignItems = "center"
            toolDescription.appendChild(container)
        }
    },
    "AI Source separation": {
        taskId: "ass",
        description: "AI audio source separation can be used to extract only the dialogue audio from an audio sample with lots of other audio mixed in, such as music, background sounds, and ambient noises. It can also remove echo from voices.",
        inputDirectory: `${window.path}/python/audio_source_separation/input`,
        outputDirectory: `${window.path}/python/audio_source_separation/output/`,
        inputFileType: ".wav"
    },
}





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


Object.keys(tools).forEach(toolName => {
    const button = createElem("button", toolName)
    button.addEventListener("click", () => {

        toolContent.style.display = "block"
        const tool = tools[toolName]

        toolNameElem.innerHTML = toolName
        toolDescription.innerHTML = tool.description

        window.tools_state.taskId = tool.taskId
        window.tools_state.inputDirectory = tool.inputDirectory
        window.tools_state.outputDirectory = tool.outputDirectory
        window.tools_state.inputFileType = tool.inputFileType

        window.tools_state.spinnerElem = toolsSpinner
        window.tools_state.progressElem = toolProgressInfo
        window.tools_state.infoElem = toolsInfo
        window.tools_state.currentFileElem = toolsCurrentFile

        if (tool.setupFn) {
            tool.setupFn(tool.taskId)
        }

    })
    toolsList.appendChild(button)
})
toolsOpenInput.addEventListener("click", () => {
    // Until I finally upgrade Electron version, to gain access to the function that implicitly does this
    const files = fs.readdirSync(window.tools_state.inputDirectory)
    if (files.length) {
        shell.showItemInFolder(`${window.tools_state.inputDirectory}/${files[0]}`)
    } else {
        shell.showItemInFolder(`${window.tools_state.inputDirectory}`)
    }
})
toolsOpenOutput.addEventListener("click", () => {
    // Until I finally upgrade Electron version, to gain access to the function that implicitly does this
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

    if (window.tools_state.spinnerElem) {
        window.tools_state.spinnerElem.style.display = "inline-block"
    }
    toolsRunTool.disabled = true
    prepAudioStart.disabled = true
    window.deleteFolderRecursive(window.tools_state.outputDirectory, true)

    window.tools_state.taskFiles = fs.readdirSync(window.tools_state.inputDirectory)

    if (window.tools_state.inputFileType) {
        window.tools_state.taskFiles = window.tools_state.taskFiles.filter(f => f.endsWith(window.tools_state.inputFileType))
    }

    doNextTaskItem()
})

const doNextTaskItem = () => {
    const inPath = `${window.tools_state.inputDirectory}/${window.tools_state.taskFiles[window.tools_state.taskFileIndex]}`

    window.tools_state.currentFileElem.innerHTML = `File: ${window.tools_state.taskFiles[window.tools_state.taskFileIndex]}`

    // const toolMeta = Object.keys(tools).find(key => tools[key].taskId==tools_state.taskId)
    // const toolSettings = toolMeta.toolSettings || {}
    const toolSettings = window.tools_state.toolSettings[window.tools_state.taskId] || {}


    window.ws.send(JSON.stringify({model: window.tools_state.taskId, task: "runTask", data: {
        outputDirectory: window.tools_state.outputDirectory,
        toolSettings,
        inPath
    }}))
}


window.websocket_handlers["task_info"] = (data) => {
    // console.log("tasks_info", data)
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

window.websocket_handlers["tasks_next"] = (data) => {
    // console.log("tasks_next")
    window.tools_state.progressElem.innerHTML = `${window.tools_state.taskFileIndex}/${window.tools_state.taskFiles.length} files done...`

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


// Preprocess audio
prepAudioStart.addEventListener("click", async () => {

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