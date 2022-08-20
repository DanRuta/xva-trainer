"use strict"
window.appVersion = "1.1.3"
app_version.innerHTML = "v"+window.appVersion
window.PRODUCTION = module.filename.includes("resources")
const path = PRODUCTION ? `${__dirname.replace(/\\/g,"/")}` : `${__dirname.replace(/\\/g,"/")}`
window.path = path
window.websocket_handlers = {}
const fs = require("fs")

window.SERVER_PORT = 8002
window.WEBSOCKET_PORT = 8001
try {
    const lines = fs.readFileSync(`${path}/ports.txt`, "utf8").split("\n")
    window.SERVER_PORT = parseInt(lines[0].split(",")[1].trim())
    window.WEBSOCKET_PORT = parseInt(lines[1].split(",")[1].trim())
} catch (e) {console.log(e)}
console.log(`Server port: ${window.SERVER_PORT}`)
console.log(`WebSocket port: ${window.WEBSOCKET_PORT}`)


window.smi = require('node-nvidia-smi')
const doFetch = require("node-fetch")
const {exec} = require("child_process")
const {shell, ipcRenderer} = require("electron")
const {xVAAppLogger} = require("./javascript/appLogger.js")
window.appLogger = new xVAAppLogger(`./app.log`, window.appVersion)
window.appLogger.log(`Ports | Server: ${window.SERVER_PORT} | WebSocket: ${window.WEBSOCKET_PORT}`)
process.on(`uncaughtException`, (data, origin) => {window.appLogger.log(`uncaughtException: ${data}`);window.appLogger.log(`uncaughtException: ${origin}`)})
window.onerror = (err, url, lineNum) => window.appLogger.log(`[line: ${lineNum}] onerror: ${err}`)
require("./javascript/util.js")
require("./javascript/settingsMenu.js")
require("./javascript/tools.js")
const execFile = require('child_process').execFile
const spawn = require("child_process").spawn

window.hasCUDA = false
if (!window.ws || !window.ws.readyState) {
    window.spinnerModal("Waiting for the WebSocket to connect...")
    const checkIsUp = () => {
        setTimeout(() => {
            if (window.ws && window.ws.readyState) {
                window.closeModal()
            } else {
                checkIsUp()
            }
        }, 500)
    }
    checkIsUp()
}
smi((err, data) => {
    if (!data || !data.nvidia_smi_log.cuda_version) {
        return window.errorModal(`CUDA installation not detected. This is required (along with an NVIDIA GPU) for most things to work. Please install it from the NVIDIA website. You can verify its installation by running "nvcc --version" on the command-line.<br>You can switch to a CPU-only installation in the settings, to use the tools, but the training is still GPU only.`)
    }
    if (err) {
        return window.errorModal(`Error reading GPU data:<br>${err}`)
    }
    window.hasCUDA = true
})

// Start the server
if (window.PRODUCTION) {
    window.pythonProcess = spawn(`${path}/cpython_${window.userSettings.installation}/server.exe`, {stdio: "ignore"})
}
ipcRenderer.send('updateDiscord', {})

const initWebSocket = () => {

    window.ws = new WebSocket(`ws://localhost:${window.WEBSOCKET_PORT}`)

    ws.onopen = event => {
        console.log("WebSocket open")
        clearInterval(initWebSocketInterval)
        if (window.hasCUDA) { // Don't clear the CUDA error message
            window.closeModal()
        }
    }

    ws.onclose = event => {
        console.log("WebSocket closed", event)
    }

    ws.onmessage = event => {
        console.log("event", event)

        if (event.data.includes("Set stage to: ")) {
            const stage = parseInt(event.data.split(": ")[1].split(" ")[0])
            window.training_state.datasetsQueue[window.training_state.trainingQueueItem].status = `Stage ${stage}`
            const allRowElems = Array.from(document.querySelectorAll("#trainingQueueRecordsContainer>div"))
            allRowElems[window.training_state.trainingQueueItem].children[2].style.background = "goldenrod"
            allRowElems[window.training_state.trainingQueueItem].children[2].style.color = "white"
            allRowElems[window.training_state.trainingQueueItem].children[2].innerHTML = window.training_state.datasetsQueue[window.training_state.trainingQueueItem].status
            window.saveTrainingQueueJSON()

            stage_select.value = `stage${stage}`
            window.training_state.stage_viewed = stage
            window.updateTrainingLogText()
            window.updateTrainingGraphs()
        } else if (event.data.includes("TRAINING_ERROR:")) {
            const errorMessage = event.data.split("TRAINING_ERROR:")[1].split("\n").join("<br>")
            window.errorModal(errorMessage).then(() => {
                trainingStopBtn.click()
            })
        } else if (event.data.includes("Finished training HiFi-GAN")) {

            window.training_state.datasetsQueue[window.training_state.trainingQueueItem].status = `Finished`
            const allRowElems = Array.from(document.querySelectorAll("#trainingQueueRecordsContainer>div"))
            allRowElems[window.training_state.trainingQueueItem].children[2].style.background = "green"
            allRowElems[window.training_state.trainingQueueItem].children[2].style.color = "white"
            allRowElems[window.training_state.trainingQueueItem].children[2].innerHTML = `Finished`
            window.saveTrainingQueueJSON()

            trainingStopBtn.click()

            if (window.training_state.isBatchTraining) {

                if (window.training_state.trainingQueueItem<(window.training_state.datasetsQueue.length-1)) {

                    // Go down the list, until the first item is found that is not yet finished
                    let itemIndex = window.training_state.trainingQueueItem
                    do {
                        if (window.training_state.datasetsQueue[itemIndex].status!="Finished") {
                            break
                        }
                        itemIndex++

                    } while (itemIndex<window.training_state.datasetsQueue.length)

                    if (itemIndex>=window.training_state.datasetsQueue.length) {
                        // Batch training is finished
                        window.training_state.isBatchTraining = false

                    } else {
                        window.training_state.isBatchTraining = true
                        window.training_state.trainingQueueItem = itemIndex
                        allRowElems[window.training_state.trainingQueueItem].click()
                        trainingStartBtn.click()
                    }

                } else {
                    window.training_state.isBatchTraining = false
                }
            }

        } else if (event.data.includes("ERROR")) {

            window.appLogger.log(event.data)
            window.errorModal(event.data).then(() => {

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

        } else {
            try {
                const response = event.data ? JSON.parse(event.data) : undefined
                console.log("response", response)

                if (Object.keys(window.websocket_handlers).includes(response["key"])) {
                    window.websocket_handlers[response["key"]](response["data"])
                }
            } catch (e) {
                if (window.tools_state.running) {
                    window.errorModal(event.data.split("\n").join("<br>"))
                }
            }
        }
    }
}

const initWebSocketInterval = setInterval(initWebSocket, 1000)



window.datasets = {}
window.filteredRows = undefined
window.appState = {
    currentDataset: undefined,
    recordFocus: undefined,
    paginationIndex: 0,
    skipRefreshing: false
}
window.recordingState = {}
window.watchedModelsDirs = {}
setTimeout(() => {
    window.electron = require("electron")
}, 1000)



window.saveDatasetToFile = (datasetName) => {
    const metadata_out = []
    window.datasets[datasetName].metadata.forEach(row => {
        metadata_out.push(`${row[0].fileName}|${row[0].text.replace(/\r\n/g,"").replace(/\n/g,"")}`)
    })
    fs.writeFileSync(`${window.userSettings.datasetsPath}/${datasetName}/metadata.csv`, metadata_out.join("\n"))
}


let refreshTimer
let refreshButton
let refreshFn = () => {
    if (refreshButton) {
        if (!window.appState.skipRefreshing && trainContainer.style.display=="none") {
            refreshButton.click()
        }
        window.appState.skipRefreshing = false
        refreshButton = undefined
    }
}
let startRefreshTimer = () => {
    if (refreshTimer != null) {
        clearTimeout(refreshTimer)
        refreshTimer = null
    }
    refreshTimer = setTimeout(refreshFn, 500)
}
window.refreshDatasets = () => {
    try {
        const datasetFolders = fs.readdirSync(window.userSettings.datasetsPath).filter(name => !name.includes(".") && (fs.existsSync(`${window.userSettings.datasetsPath}/${name}/metadata.csv`) || fs.existsSync(`${window.userSettings.datasetsPath}/${name}/wavs`)))
        const buttons = []
        datasetFolders.forEach(dataset => {
            window.datasets[dataset] = {metadata: []}

            const button = createElem("div.voiceType", dataset)

            button.addEventListener("click", event => {

                window.appState.skipRefreshing = false

                Array.from(document.querySelectorAll(".mainPageButton")).forEach(button => button.disabled = false)

                title.innerHTML = `Dataset: ${dataset}`

                const records = []
                const metadataAudioFiles = []
                window.appState.recordFocus = undefined

                if (fs.existsSync(`${window.userSettings.datasetsPath}/${dataset}/metadata.csv`)) {
                    const data = fs.readFileSync(`${window.userSettings.datasetsPath}/${dataset}/metadata.csv`, "utf8")
                    if (data.length) {
                        data.split("\n").forEach(line => {
                            if (line.trim().length==0) return
                            metadataAudioFiles.push(line.split("|")[0])
                            records.push({
                                fileName: line.split("|")[0],
                                text: line.split("|")[1].replace(/\r\n/g,"").replace(/\n/g,"")
                            })
                        })
                    }
                }

                // Load stray audio files
                if (fs.existsSync(`${window.userSettings.datasetsPath}/${dataset}/wavs`)) {
                    const audioFiles = fs.readdirSync(`${window.userSettings.datasetsPath}/${dataset}/wavs`)

                    const strayFiles = audioFiles.filter(fname => !metadataAudioFiles.includes(fname))
                    strayFiles.forEach(fname => {
                        records.push({
                            fileName: fname,
                            text: ""
                        })
                    })
                }

                window.appState.currentDataset = dataset

                const numPages = Math.ceil(records.length/window.userSettings.paginationSize)
                ofTotalPages.innerHTML = `of ${numPages}`
                pageNum.value = 1

                if (records.length) {
                    batch_main.style.display = "block"
                    batchDropZoneNote.style.display = "none"
                    batchRecordsHeader.style.display = "flex"
                    populateRecordsList(dataset, records)
                    refreshRecordsList(dataset)
                } else {
                    batch_main.style.display = "flex"
                    batchDropZoneNote.style.display = "flex"
                    batchRecordsHeader.style.display = "none"
                    batchRecordsContainer.innerHTML = ""
                }

                if (!fs.existsSync(`${window.userSettings.datasetsPath}/${dataset}/dataset_metadata.json`)) {
                    addmissingmeta_btn_container.style.display = "flex"
                    editdatasetmeta_btn_container.style.display = "none"
                } else {
                    editdatasetmeta_btn_container.style.display = "flex"
                    addmissingmeta_btn_container.style.display = "none"
                }

                // Check for duplicates
                const duplicateLines = []
                const duplicateFileNames = new Set()
                const existingFilesReferenced = new Set()
                records.forEach(record => {
                    if (existingFilesReferenced.has(record.fileName)) {
                        duplicateFileNames.add(record.fileName)
                    }
                    existingFilesReferenced.add(record.fileName)
                })
                records.forEach(record => {
                    if (duplicateFileNames.has(record.fileName)) {
                        duplicateLines.push(record)
                    }
                })

                if (duplicateLines.length) {
                    totalDataDuplicates.innerHTML = duplicateLines.length
                    uniqueDataDuplicates.innerHTML = duplicateFileNames.size
                    datasetDuplicatesWarning.style.display = "flex"
                } else {
                    datasetDuplicatesWarning.style.display = "none"
                }

            })
            buttons.push(button)

            if (!fs.existsSync(`${window.userSettings.datasetsPath}/${dataset}`)) {
                fs.mkdirSync(`${window.userSettings.datasetsPath}/${dataset}`)
            }

            if (!Object.keys(window.watchedModelsDirs).includes(`${window.userSettings.datasetsPath}/${dataset}`)) {
                window.watchedModelsDirs[`${window.userSettings.datasetsPath}/${dataset}`] = fs.watch(`${window.userSettings.datasetsPath}/${dataset}`, {recursive: true, persistent: true}, (eventType, fileName) => {
                    if (dataset==window.appState.currentDataset && !window.appState.skipRefreshing) {
                        refreshButton = button
                        startRefreshTimer()
                    }
                })
            }

        })
        voiceTypeContainer.innerHTML = ""
        buttons.sort((a,b) => a.innerHTML<b.innerHTML?-1:1).forEach(button => voiceTypeContainer.appendChild(button))

    } catch (e) {
        console.log(e)
    }
}
window.refreshDatasets()






// Recording panel functionality
// =============================
window.clearRecordFocus = ri => {
    if (ri!=undefined) {
        batchRecordsContainer.children[ri].style.background = "rgb(75,75,75)"
        batchRecordsContainer.children[ri].style.color = "white"
        Array.from(batchRecordsContainer.children[ri].children).forEach(c => {
            c.style.color = "white"
        })
    }
}
window.setRecordFocus = ri => {
    if (window.appState.recordFocus!=undefined) {
        window.clearRecordFocus(window.appState.recordFocus%window.userSettings.paginationSize)
    }

    if (ri!=undefined) {
        batchRecordsContainer.children[ri%window.userSettings.paginationSize]
        batchRecordsContainer.children[ri%window.userSettings.paginationSize].style.background = "rgb(255,255,255)"
        batchRecordsContainer.children[ri%window.userSettings.paginationSize].style.color = "black"
        Array.from(batchRecordsContainer.children[ri%window.userSettings.paginationSize].children).forEach(c => {
            c.style.color = "black"
        })
        textInput.value = window.filteredRows[ri][0].text
        outFileNameInput.value = window.filteredRows[ri][0].fileName
    } else {
        textInput.value = ""
        outFileNameInput.value = ""
    }
    window.appState.recordFocus = ri
}
right.addEventListener("click", event => {
    if (event.target.nodeName=="BUTTON" || event.target.nodeName=="INPUT" || event.target.nodeName=="SVG" || event.target.nodeName=="IMG" || event.target.nodeName=="TEXTAREA" || event.target.nodeName=="path" ||
        ["row", "rowItem"].includes(event.target.className))  {
        return
    }
    if (window.appState.currentDataset && window.appState.recordFocus!=undefined && window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][0].text.trim().toLowerCase().replace(/\n/, "")!=textInput.value.trim().toLowerCase().replace(/\n/, "")) {
        window.confirmModal("The transcript for this line has changed discard changes?").then(resp => {
            if (resp) {
                clearRecordFocus(window.appState.recordFocus%window.userSettings.paginationSize)
            }
        })
    } else {
        if (window.appState.recordFocus!=undefined) {
            clearRecordFocus(window.appState.recordFocus%window.userSettings.paginationSize)
        }
    }
})
// =============================





// Button functionalities
// ======================
btn_save.addEventListener("click", () => {
    if (window.appState.recordFocus!=undefined) {

        const doTheRest = () => {
            window.appState.skipRefreshing = true
            // Rename audio file if one exists
            const oldFileName = window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][0].fileName
            const newFileName = outFileNameInput.value.replace(".wav","")+".wav"
            if (oldFileName!=newFileName && fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${oldFileName}`)) {

                if (fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${newFileName}`)) {
                    errorModal(`This file already exists. Please choose a different name.<br><br><i>${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${newFileName}</i>`)
                    return
                }

                fs.renameSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${oldFileName}`, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${newFileName}`)
            }

            window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][0].text = textInput.value.replace(/\|/g,"").replace(/\n/g,"").replace(/\r/g,"").trim()
            window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][0].fileName = outFileNameInput.value.replace(".wav","")+".wav"
            window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][1].children[2].innerHTML = outFileNameInput.value.replace(".wav","")+".wav"
            window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][1].children[3].innerHTML = textInput.value.replace(/\|/g,"").replace(/\n/g,"").replace(/\r/g,"").trim()

            window.saveDatasetToFile(window.appState.currentDataset)

            if (window.appState.recordFocus+1<batchRecordsContainer.childElementCount) {
                setRecordFocus(window.appState.recordFocus+1)
            } else {
                setRecordFocus()
            }
            window.appState.skipRefreshing = false
        }

        // Copy over the recorded file, if one exists
        if (window.recordingState.tempAudioFileName && fs.existsSync(window.recordingState.tempAudioFileName)) {
            const targetOutput = outFileNameInput.value.replace(".wav","")+".wav"
            if (fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${targetOutput}`)) {
                errorModal(`This file already exists. Please choose a different name.<br><br><i>${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${targetOutput}</i>`)

                confirmModal("An audio file already exists. Are you sure you'd like to overwrite it with the newer one?").then(confirmation => {
                    if (confirmation) {
                        fs.copyFileSync(window.recordingState.tempAudioFileName, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${targetOutput}`)
                        wave_container.innerHTML = ""
                        clearTempFiles()
                        doTheRest()
                    }
                })
            } else {
                fs.copyFileSync(window.recordingState.tempAudioFileName, `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${targetOutput}`)
                wave_container.innerHTML = ""
                clearTempFiles()
                doTheRest()
            }
        } else {
            doTheRest()
        }

    } else {
        if (textInput.value.length || outFileNameInput.value.length) {
            window.appState.skipRefreshing = true
            const currentText = textInput.value.replace(/\|/g,"") || ""
            const currentFileName = outFileNameInput.value || getNextFileName(window.appState.currentDataset)
            btn_newline.click()
            textInput.value = currentText
            outFileNameInput.value = currentFileName
            setTimeout(() => btn_save.click(), 0)
        }
    }
})
btn_newline.addEventListener("click", () => {
    if (window.appState.currentDataset!=undefined) {
        const newRecord = {}
        newRecord.fileName = getNextFileName(window.appState.currentDataset)+".wav"
        newRecord.text = ""

        const recordIndex = window.datasets[window.appState.currentDataset].metadata.length

        const row = createElem("div.row")
        const rNumElem = createElem("div.rowItem", recordIndex.toString())
        const rGameElem = createElem("div.rowItem", "")
        const rTextElem = createElem("div.rowItem", "")
        const rOutPathElem = createElem("div.rowItem", "&lrm;"+newRecord.fileName+"&lrm;")


        row.appendChild(rNumElem)
        row.appendChild(rGameElem)
        row.appendChild(rOutPathElem)
        row.appendChild(rTextElem)
        row.addEventListener("click", event => {
            setRecordFocus(recordIndex)
        })


        window.datasets[window.appState.currentDataset].metadata.push([newRecord, row, recordIndex])
        batch_main.style.display = "block"
        batchDropZoneNote.style.display = "none"
        batchRecordsHeader.style.display = "flex"
        refreshRecordsList(window.appState.currentDataset)
        setRecordFocus(recordIndex)
        textInput.focus()
    }
})
window.playRecordAudio = ri => {
    if (window.appState.currentDataset==undefined) return

    const audioPath = `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${window.datasets[window.appState.currentDataset].metadata[ri][0].fileName}`
    if (fs.existsSync(audioPath)) {
        const audioPreview = createElem("audio", {autoplay: false}, createElem("source", {src: audioPath}))
    }
}
btn_play.addEventListener("click", () => {
    if (window.appState.recordFocus!=undefined) {
        if (wave_container.children.length) {
            wave_container.children[0].play()
        } else {
            window.playRecordAudio(window.appState.recordFocus)
        }
    }
})
btn_delete.addEventListener("click", () => {
    if (window.appState.recordFocus!=undefined) {

        const text = window.filteredRows[window.appState.recordFocus][0].text
        const recordFocusIndex = window.datasets[window.appState.currentDataset].metadata.findIndex(record => record[3]==window.filteredRows[window.appState.recordFocus][3])

        confirmModal(`Are you sure you'd like to delete this line?<br><br><i>${text}</i>`).then(confirmation => {
            if (confirmation) {
                const fileName = window.datasets[window.appState.currentDataset].metadata[recordFocusIndex][0].fileName
                delete window.datasets[window.appState.currentDataset].metadata[recordFocusIndex]
                window.saveDatasetToFile(window.appState.currentDataset)

                if (fileNameSearch.value!="%duplicates%" && fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${fileName}`)) {
                    fs.unlinkSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${fileName}`)
                }
            }
        })
    }
    window.appState.recordFocus = undefined
})
btn_opendatasetfolder.addEventListener("click", () => {
    if (window.appState.currentDataset!=undefined) {
        console.log(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`)
        shell.showItemInFolder(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/metadata.csv`)
    }
})
btn_deletedataset.addEventListener("click", () => {
    if (window.appState.currentDataset!=undefined) {
        confirmModal(`Are you VERY sure you'd like to completely delete this dataset? This cannot be undone. Dataset:<br><br>${window.appState.currentDataset}`).then(confirmation => {
            if (confirmation) {
                // window.datasets[window.appState.currentDataset].metadata = []
                // saveDatasetToFile(window.appState.currentDataset)
                window.watchedModelsDirs[`${window.userSettings.datasetsPath}/${window.appState.currentDataset}`].close()
                batch_main.style.display = "flex"
                batchDropZoneNote.style.display = "flex"
                batchRecordsHeader.style.display = "none"
                batchRecordsContainer.innerHTML = ""

                window.deleteFolderRecursive(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}`)
                window.appState.currentDataset = undefined
                title.innerHTML = "Select dataset"
                window.refreshDatasets()
                addmissingmeta_btn_container.style.display = "none"
                editdatasetmeta_btn_container.style.display = "none"
            }
        })
    }
})
btn_deleteall.addEventListener("click", () => {
    if (window.appState.currentDataset!=undefined) {
        confirmModal(`Are you VERY sure you'd like to completely delete everyhing for this dataset? This cannot be undone. Dataset:<br><br>${window.appState.currentDataset}`).then(confirmation => {
            if (confirmation) {
                window.datasets[window.appState.currentDataset].metadata = []
                saveDatasetToFile(window.appState.currentDataset)
            }
        })
    }
})
btn_autotranscribe.addEventListener("click", () => {
    if (window.appState.currentDataset!=undefined) {
        confirmModal(`Are you sure you'd like to kick off the auto-transcription process?<br>This will run for all 22050Hz audio with no text transcript.`).then(confirmation => {
            if (confirmation) {
                setTimeout(() => {
                    createModal("spinner", "Auto-transcribing...<br>This may take a few minutes if there are hundreds of lines.<br>Audio files must be mono 22050Hz<br><br>This window will close if there is an error.")
                    window.appState.skipRefreshing = true


                    const inputDirectory =  `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`

                    window.tools_state.taskId = "transcribe"
                    window.tools_state.inputDirectory = inputDirectory
                    window.tools_state.outputDirectory = ""
                    window.tools_state.inputFileType = "folder"

                    window.tools_state.spinnerElem = toolsSpinner
                    window.tools_state.progressElem = toolProgressInfo
                    window.tools_state.infoElem = toolsInfo
                    window.tools_state.currentFileElem = toolsCurrentFile

                    window.tools_state.post_callback = () => {
                        window.appState.skipRefreshing = false
                        window.refreshRecordsList(window.appState.currentDataset)
                        closeModal()
                    }

                    toolsRunTool.click()

                }, 300)
            }
        })
    }
})
btn_record.addEventListener("click", () => {
    if (window.appState.currentDataset!=undefined) {
        if (window.recordingState.isReadingMic) {
            window.stopRecord()
        } else {
            wave_container.innerHTML = ""
            btn_record.style.background = "red"
            window.startRecord()
        }
    }
})
btn_clearRecord.addEventListener("click", () => {
    btn_clearRecord.style.display = "none"
    wave_container.innerHTML = ""
    // try {
        // fs.unlinkSync(`${window.path}/${window.recordingState.tempAudioFileName}`)
        fs.unlinkSync(window.recordingState.tempAudioFileName)
    // } catch (e) {}
})
// ======================



window.clearTempFiles = () => {
    const files = fs.readdirSync(window.path).filter(fname => fname.includes("recorded_file_"))
    files.forEach(fileName => {
        fs.unlinkSync(`${window.path}/${fileName}`)
    })
}
clearTempFiles()


// Text file loading
// =================
// https://stackoverflow.com/questions/1293147/example-javascript-code-to-parse-csv-data
const readFile = (file) => {
    return new Promise((resolve, reject) => {
        const dataLines = []
        const reader = new FileReader()
        reader.readAsText(file)
        reader.onloadend = () => {
            const lines = reader.result.split("\n")
            const header = ["fileName", "text", "text"]
            // const header = lines.shift().split(",").map(head => head.replace(/\r/, ""))
            lines.forEach(line => {
                const record = {}
                if (line.trim().length) {
                    const parts = CSVToArray(line, "|")[0]
                    parts.forEach((val, vi) => {
                        record[header[vi].replace(/^"/, "").replace(/"$/, "")] = val
                    })
                    dataLines.push(record)
                }
            })
            resolve(dataLines)
        }
    })
}
const readFileTxt = (file) => {
    return new Promise((resolve, reject) => {
        const dataLines = []
        const reader = new FileReader()
        reader.readAsText(file)
        reader.onloadend = () => {
            const lines = reader.result.replace(/\r\n/g, "\n").split("\n")
            lines.forEach(line => {
                if (line.trim().length) {
                    const record = {}
                    // record.game_id = window.currentModel.games[0].gameId
                    record.fileName = undefined //dataLines.length
                    record.text = line
                    dataLines.push(record)
                }
            })
            resolve(dataLines)
        }
    })
}

const uploadBatchCSVs = async (eType, event) => {

    if (["dragenter", "dragover"].includes(eType)) {
        batch_main.style.background = "#5b5b5b"
        batch_main.style.color = "white"
    }
    if (["dragleave", "drop"].includes(eType)) {
        batch_main.style.background = "#4b4b4b"
        batch_main.style.color = "gray"
    }

    event.preventDefault()
    event.stopPropagation()

    const dataLines = []

    if (eType=="drop") {

        batchDropZoneNote.innerHTML = "Processing data"

        const dataTransfer = event.dataTransfer
        const files = Array.from(dataTransfer.files)
        for (let fi=0; fi<files.length; fi++) {
            const file = files[fi]
            if (!file.name.endsWith(".csv")) {
                if (file.name.endsWith(".txt")) {
                    // if (window.currentModel) {
                    const records = await readFileTxt(file)
                    records.forEach(item => dataLines.push(item))
                    // }
                    continue
                } else {
                    continue
                }
            }

            const records = await readFile(file)
            records.forEach(item => {
                dataLines.push(item)
            })
        }

        const cleanedData = dataLines
        batch_main.style.display = "block"
        batchDropZoneNote.style.display = "none"
        batchRecordsHeader.style.display = "flex"
        // batch_clearBtn.style.display = "inline-block"


        if (cleanedData.length) {
            populateRecordsList(window.appState.currentDataset, cleanedData, true)
            refreshRecordsList(window.appState.currentDataset)
            saveDatasetToFile(window.appState.currentDataset)
        } else {
            // batch_clearBtn.click()
        }

        batchDropZoneNote.innerHTML = "Drag and drop .txt or .csv files here"
    }
}
window.searchDatasetRows = rows => {

    const fileNameQuery = fileNameSearch.value.trim().toLowerCase()
    const transcriptQuery = transcriptSearch.value.trim().toLowerCase()

    const filteredRows = []

    if (fileNameQuery=="%duplicates%") {
        const duplicateLines = new Set()
        const existingFilesReferenced = new Set()
        rows.forEach(record => {
            if (existingFilesReferenced.has(record[0].fileName)) {
                duplicateLines.add(record[0].fileName)
            }
            existingFilesReferenced.add(record[0].fileName)
        })
        rows.forEach(record => {
            if (duplicateLines.has(record[0].fileName)) {
                record[2] = filteredRows.length
                filteredRows.push(record)
            }
        })
    } else {
        rows.forEach(row => {
            if (row[0].fileName.toLowerCase().includes(fileNameQuery) && row[0].text.toLowerCase().includes(transcriptQuery)) {
                row[2] = filteredRows.length
                filteredRows.push(row)
            }
        })
    }
    return filteredRows
}
fileNameSearch.addEventListener("keyup", () => {
    window.refreshRecordsList()
})
transcriptSearch.addEventListener("keyup", () => {
    window.refreshRecordsList()
})


window.refreshRecordsList = (dataset) => {
    if (window.tools_state.running || window.appState.skipRefreshing || !window.appState.currentDataset) {
        return
    }
    batchRecordsContainer.innerHTML = ""

    const filteredRows = window.searchDatasetRows(window.datasets[window.appState.currentDataset].metadata)
    window.filteredRows = filteredRows
    const numPages = Math.ceil(window.filteredRows.length/window.userSettings.paginationSize)
    ofTotalPages.innerHTML = `of ${numPages}`

    doFetch(`http://localhost:${window.SERVER_PORT}/getAudioLengthOfDir`, {
        method: "Post",
        body: JSON.stringify({
            directory: `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs`
        })
    }).then(r=>r.text()).then((res) => {

        let displayString = `${filteredRows.length} files`
        if (res) {

            let meanSeconds = parseInt(res.split("|")[0])
            let totalSeconds = parseInt(res.split("|")[1])

            displayString += " | "+window.formatSecondsToTimeString(totalSeconds)
        }
        totalRecords.innerHTML = displayString
    })

    totalRecords.innerHTML = `${filteredRows.length} files`


    const startIndex = (window.appState.paginationIndex*window.userSettings.paginationSize)
    const endIndex = Math.min(startIndex+window.userSettings.paginationSize, filteredRows.length)

    for (let ri=startIndex; ri<endIndex; ri++) {
        const recordAndElem = filteredRows[ri]

        if (window.wer_cache[window.appState.currentDataset]) {
            const score = window.wer_cache[window.appState.currentDataset][ri]
            const r_col = Math.min(score, 1)
            const g_col = 1 - r_col
            recordAndElem[1].children[4].style.background = `rgba(${r_col*255},${g_col*255},50, 0.7)`
        } else {
            recordAndElem[1].children[4].style.background = `none`
        }

        fs.exists(recordAndElem[3], exists => {
            if (exists) {
                const audio = createElem("audio", {controls: true}, createElem("source", {
                    src: recordAndElem[3],
                    type: `audio/wav`
                }))
                audio.style.zIndex = "1"

                recordAndElem[1].children[1].innerHTML = ""
                audio.addEventListener("canplaythrough", () => {
                    if (recordAndElem[1].children[1].innerHTML == "") {
                        recordAndElem[1].children[1].appendChild(audio)
                    }
                })
            }
        })

        recordAndElem[1].children[0].innerHTML = (ri+1)
        recordAndElem[1].addEventListener("click", event => {
            setRecordFocus(filteredRows[ri][2])
        })
        batchRecordsContainer.appendChild(recordAndElem[1])
    }
}


const populateRecordsList = (dataset, records, additive=false) => {

    if (!additive) {
        window.datasets[dataset].metadata = []
    }

    batchDropZoneNote.style.display = "none"

    const allPaths = fs.readdirSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/`)

    records.forEach((record, ri) => {
        const row = createElem("div.row")
        const rNumElem = createElem("div.rowItem")
        const rGameElem = createElem("div.rowItem", "")

        record.fileName = record.fileName==undefined ? getNextFileName(dataset)+".wav" : record.fileName
        const rOutPathElem = createElem("div.rowItem", record.fileName)
        rOutPathElem.title = record.fileName
        const rTextElem = createElem("div.rowItem", record.text)
        rTextElem.title = record.text


        row.appendChild(rNumElem)
        const potentialAudioPath = `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${record.fileName}`
        if (allPaths.includes(record.fileName)) {
            rGameElem.style.zIndex = "1"
        }
        row.appendChild(rGameElem)
        row.appendChild(rOutPathElem)
        row.appendChild(rTextElem)

        const wer_elem = createElem("div.wer")
        if (record.score) {
            const r_col = Math.min(record.score, 1)
            const g_col = 1 - r_col
            wer_elem.style.background = `rgba(${r_col*255},${g_col*255},50, 0.7)`
        }

        row.appendChild(wer_elem)

        window.datasets[dataset].metadata.push([record, row, ri, potentialAudioPath])
    })
}

batch_main.addEventListener("dragenter", event => uploadBatchCSVs("dragenter", event), false)
batch_main.addEventListener("dragleave", event => uploadBatchCSVs("dragleave", event), false)
batch_main.addEventListener("dragover", event => uploadBatchCSVs("dragover", event), false)
batch_main.addEventListener("drop", event => uploadBatchCSVs("drop", event), false)

paginationPrev.addEventListener("click", () => {
    pageNum.value = Math.max(1, parseInt(pageNum.value)-1)
    window.appState.paginationIndex = pageNum.value-1
    if (window.appState.recordFocus!=undefined) {
        window.clearRecordFocus(window.appState.recordFocus%window.userSettings.paginationSize)
    }
    window.appState.recordFocus = undefined
    refreshRecordsList(window.appState.currentDataset)
})
paginationNext.addEventListener("click", () => {
    const numPages = Math.ceil(window.filteredRows.length/window.userSettings.paginationSize)
    pageNum.value = Math.min(parseInt(pageNum.value)+1, numPages)
    if (window.appState.recordFocus!=undefined) {
        window.clearRecordFocus(window.appState.recordFocus%window.userSettings.paginationSize)
    }
    window.appState.recordFocus = undefined
    window.appState.paginationIndex = pageNum.value-1
    refreshRecordsList(window.appState.currentDataset)
})
pageNum.addEventListener("change", () => {
    const numPages = Math.ceil(window.filteredRows.length/window.userSettings.paginationSize)
    pageNum.value = Math.max(1, Math.min(parseInt(pageNum.value), numPages))
    if (window.appState.recordFocus!=undefined) {
        window.clearRecordFocus(window.appState.recordFocus%window.userSettings.paginationSize)
    }
    window.appState.recordFocus = undefined
    window.appState.paginationIndex = pageNum.value-1
    refreshRecordsList(window.appState.currentDataset)
})
setting_paginationSize.addEventListener("change", () => {
    const numPages = Math.ceil(window.filteredRows.length/window.userSettings.paginationSize)
    pageNum.value = Math.max(1, Math.min(parseInt(pageNum.value), numPages))
    window.appState.paginationIndex = pageNum.value-1
    ofTotalPages.innerHTML = `of ${numPages}`
    if (window.appState.recordFocus!=undefined) {
        window.clearRecordFocus(window.appState.recordFocus%window.userSettings.paginationSize)
    }
    window.appState.recordFocus = undefined

    refreshRecordsList(window.appState.currentDataset)
})
// =================








// Recording
// =========
window.recordingState = {
    isReadingMic: false,
    elapsedRecording: 0,
    recorder: null,
    stream: null
}
// Populate the microphone dropdown with the available options
navigator.mediaDevices.enumerateDevices().then(devices => {
    devices = devices.filter(device => device.kind=="audioinput" && device.deviceId!="default" && device.deviceId!="communications")
    devices.forEach(device => {
        const option = createElem("option", device.label)
        option.value = device.deviceId
        setting_mic_selection.appendChild(option)
    })

    setting_mic_selection.addEventListener("change", () => {
        window.userSettings.microphone = setting_mic_selection.value
        window.saveUserSettings()
        window.initMic()
    })

    if (Object.keys(window.userSettings).includes("microphone")) {
        setting_mic_selection.value = window.userSettings.microphone
    } else {
        window.userSettings.microphone = setting_mic_selection.value
        window.saveUserSettings()
    }
})
window.initMic = () => {
    return new Promise(resolve => {
        const deviceId = window.userSettings.microphone

        if (window.recordingState.context!=undefined) {
            window.recordingState.context.close()
        }

        navigator.mediaDevices.getUserMedia({audio: {deviceId: deviceId}}).then(stream => {
            // const audio_context = new AudioContext({sampleRate: 22050})
            const audio_context = new AudioContext()
            const input = audio_context.createMediaStreamSource(stream)
            window.recordingState.stream = stream
            window.recordingState.audio_context = audio_context
            // window.recordingState.recorder = new Recorder(input, {numChannels: 1})

            const dest = audio_context.createMediaStreamDestination()
            const gainNode = audio_context.createGain()
            input.connect(gainNode)
            gainNode.connect(dest)
            gainNode.gain.value = parseInt(window.userSettings.micVolume)/100

            window.recordingState.recorder = new Recorder(gainNode, {numChannels: 1})
            window.recordingState.context = audio_context

            resolve()

        }).catch(err => {
            console.log(err)
            resolve()
        })
    })
}
window.initMic()

window.startRecord = async () => {
    await window.initMic()
    window.recordingState.recorder.record()

    window.recordingState.isReadingMic = true
    window.recordingState.elapsedRecording = Date.now()
}
window.stopRecord = (cancelled) => {

    window.recordingState.recorder.stop()
    window.recordingState.stream.getAudioTracks()[0].stop()

    if (!cancelled) {
        btn_clearRecord.style.display = "block"

        btn_record.style.background = "#ccc"
        const tempFileNum = `${Math.random().toString().split(".")[1]}`
        const fileName = `${window.path}/recorded_file_${tempFileNum}.wav`
        window.recordingState.tempAudioFileName = fileName
        outputRecording(window.recordingState.tempAudioFileName, () => {


            const doTheRest = () => {
                const audioElem = createElem("audio", {controls: true}, createElem("source", {
                    src: window.recordingState.tempAudioFileName,
                    type: `audio/wav`
                }))

                wave_container.appendChild(audioElem)

                console.log("done outputting file", window.recordingState.tempAudioFileName)
            }

            // Remove background noise
            if (window.userSettings.removeNoise) {

                const sox_path = (window.PRODUCTION ? "./resources/app" :  ".") + "/python/sox/sox.exe"
                const outputNameSox = `${window.recordingState.tempAudioFileName.replace(".wav", "")}_sil.wav`
                const soxCommand = `${sox_path} "${window.recordingState.tempAudioFileName}" "${outputNameSox}" noisered ${window.path}/noise_profile_file ${window.userSettings.noiseRemStrength} rate 22050`
                exec(soxCommand, async (err, stdout, stderr) => {
                    if (err) {
                        console.log(err)
                        console.log(stderr)
                        window.errorModal(`Error using sox to remove background noise<br><br>${stderr}`)
                    } else {
                        fs.copyFileSync(outputNameSox, window.recordingState.tempAudioFileName, fs.constants.COPYFILE_FICLONE_FORCE)
                        fs.unlinkSync(outputNameSox)
                        doTheRest()
                    }
                })
            } else {
                doTheRest()
            }
        })
    }

    window.recordingState.isReadingMic = false
    window.recordingState.elapsedRecording = 0
    // clearProgress()
}
const outputRecording = (outPath, callback) => {
    window.recordingState.recorder.exportWAV(AudioBLOB => {
        const fileReader = new FileReader()
        fileReader.onload = function() {

            if (fs.existsSync(outPath)) {
                fs.unlinkSync(outPath)
            }

            const bufferArray = Buffer.from(new Uint8Array(this.result))
            fs.writeFileSync(outPath, bufferArray)

            if (callback) {
                callback()
            }
        }
        fileReader.readAsArrayBuffer(AudioBLOB)
        window.recordingState.recorder.clear()
    })
}
// function downsampleBuffer(buffer, rate) {
//     if (rate == window.recordingState.audio_context.sampleRate) {
//         return buffer;
//     }
//     if (rate > window.recordingState.audio_context.sampleRate) {
//         throw "downsampling rate show be smaller than original sample rate";
//     }
//     var sampleRateRatio = window.recordingState.audio_context.sampleRate / rate;
//     var newLength = Math.round(buffer.length / sampleRateRatio);
//     var result = new Float32Array(newLength);
//     var offsetResult = 0;
//     var offsetBuffer = 0;
//     while (offsetResult < result.length) {
//         var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
//          // Use average value of skipped samples
//         var accum = 0, count = 0;
//         for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
//             accum += buffer[i];
//             count++;
//         }
//         result[offsetResult] = accum / count;
//         // Or you can simply get rid of the skipped samples:
//         // result[offsetResult] = buffer[nextOffsetBuffer];
//         offsetResult++;
//         offsetBuffer = nextOffsetBuffer;
//     }
//     return result;
// }

// =========



recordNoiseButton.addEventListener("click", () => {
    if (window.recordingState.isReadingMic) {
        // window.stopRecord()
        window.recordingState.recorder.stop()
        window.recordingState.stream.getAudioTracks()[0].stop()
        recordNoiseButton.style.background = "#ccc"

        const fileName = `${window.path}/recorded_noise.wav`
        try {
            fs.unlinkSync(fileName)
        } catch (e) {}
        outputRecording(fileName, () => {

            try {
                fs.unlinkSync(`${window.path}/noise_profile_file`)
            } catch (e) {}

            // Create the noise profile
            const soxCommand = `sox "${fileName}" -n noiseprof ${window.path}/noise_profile_file`
            exec(soxCommand, async (err, stdout, stderr) => {
                if (err) {
                    console.log(err)
                    console.log(stderr)
                    window.errorModal(`Error using sox to make a background noise profile<br><br>${stderr}`)
                } else {
                    const audioElem = createElem("audio", {controls: true}, createElem("source", {
                        src: fileName,
                        type: `audio/wav`
                    }))

                    noiseContainer.appendChild(audioElem)
                }
            })
        })

    } else {
        noiseContainer.innerHTML = ""
        recordNoiseButton.style.background = "red"
        window.startRecord()
    }
})
if (fs.existsSync(`${window.path}/recorded_noise.wav`)) {
    const audioElem = createElem("audio", {controls: true}, createElem("source", {
        src: `${window.path}/recorded_noise.wav`,
        type: `audio/wav`
    }))
    noiseContainer.appendChild(audioElem)
}














// == Dataset Metadata =
// =====================

const initDatasetMeta = (callback) => {
    const finishButton = createElem("button", "Submit")
    datasetMetaFinishButtonContainer.innerHTML = ""
    datasetMetaFinishButtonContainer.appendChild(finishButton)

    voiceIDInputChanged = false

    finishButton.addEventListener("click", () => {
        console.log("finish")

        if (!datasetMeta_voiceName.value.trim().length) {
            return window.createModal("error", "Please enter a voice name").then(() => closeModal(undefined, datasetMetaContainer))
        }
        if (!datasetMeta_gameId.value.trim().length) {
            return window.createModal("error", "Please enter a game ID").then(() => closeModal(undefined, datasetMetaContainer))
        }
        if (!datasetMeta_gameIdCode.value.trim().length) {
            return window.createModal("error", "Please enter a game ID code").then(() => closeModal(undefined, datasetMetaContainer))
        }
        if (!datasetMeta_voiceId.value.trim().length) {
            return window.createModal("error", "Please enter a voice ID").then(() => closeModal(undefined, datasetMetaContainer))
        }
        if (!datasetMeta_langcode.value.trim().length) {
            return window.createModal("error", "Please enter a language code").then(() => closeModal(undefined, datasetMetaContainer))
        }
        if (!datasetMeta_modelVersion.value.trim().length) {
            return window.createModal("error", "Please enter a model version").then(() => closeModal(undefined, datasetMetaContainer))
        }

        if (callback) {
            callback()
        }

        try {
            fs.makedirSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset||composedVoiceId.innerHTML}`)
            fs.mkdirSync(`${window.userSettings.datasetsPath}/${composedVoiceId.innerHTML}`)
            fs.mkdirSync(`${window.userSettings.datasetsPath}/${composedVoiceId.innerHTML}/wavs`)
        } catch (e) {}

        fs.writeFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset||composedVoiceId.innerHTML}/dataset_metadata.json`, JSON.stringify({
            "version": "2.0",
            "modelVersion": parseFloat(datasetMeta_modelVersion.value).toFixed(1),
            "modelType": "FastPitch1.1",
            "author": datasetMeta_author.value,
            "license": datasetMeta_license.value,
            "lang": datasetMeta_langcode.value.trim().toLowerCase(),
            "games": [
                {
                    "gameId": datasetMeta_gameId.value.trim().toLowerCase(),
                    "voiceId": composedVoiceId.innerHTML,
                    "voiceName": datasetMeta_voiceName.value.trim(),
                    "resemblyzer": [],
                    "gender": (datasetMeta_gender_male.checked ? "male" : (datasetMeta_gender_female.checked ? "female" : "other")),
                }
            ]
        }, null, 4))

        window.refreshDatasets()
        closeModal()
    })
}

let voiceIDInputChanged = false
let fixedFolderName = undefined


window.setupModal(btn_preprocessAudioButton, preprocessAudioContainer)
window.setupModal(btn_preprocessTextButton, preprocessTextContainer)
window.setupModal(btn_cleanAudioText, cleanAudioTextContainer)
window.setupModal(btn_checkTextQualityBtn, checkTextQualityContainer)
window.setupModal(datasetDuplicatesWarning, duplicateDatasetContainer)

btnShowAllDataDuplicates.addEventListener("click", () => {
    fileNameSearch.value = "%duplicates%"
    window.searchDatasetRows(window.datasets[window.appState.currentDataset].metadata)
    duplicateDatasetContainer.click()
    window.refreshRecordsList()
})
btnRemoveAllDataDuplicates.addEventListener("click", () => {

    const duplicateLines = []
    const duplicateFileNames = new Set()
    const existingFilesReferenced = new Set()
    window.datasets[window.appState.currentDataset].metadata.forEach(record => {
        if (existingFilesReferenced.has(record[0].fileName)) {
            duplicateFileNames.add(record[0].fileName)
        }
        existingFilesReferenced.add(record[0].fileName)
    })
    window.datasets[window.appState.currentDataset].metadata.forEach(record => {
        if (duplicateFileNames.has(record[0].fileName)) {
            duplicateLines.push(record)
        }
    })

    window.confirmModal(`Are you sure you'd like to delete ${duplicateLines.length} records from your dataset?`).then(resp => {
        if (resp) {
            for (let i=0; i<window.datasets[window.appState.currentDataset].metadata.length; i++) {
                if (duplicateFileNames.has(window.datasets[window.appState.currentDataset].metadata[i][0].fileName)) {
                    const fileName = window.datasets[window.appState.currentDataset].metadata[i][0].fileName
                    if (fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${fileName}`)) {
                        fs.unlinkSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${fileName}`)
                    }
                    window.datasets[window.appState.currentDataset].metadata[i] = undefined
                }
            }

            window.datasets[window.appState.currentDataset].metadata = window.datasets[window.appState.currentDataset].metadata.filter(item => item!=undefined)
            window.saveDatasetToFile(window.appState.currentDataset)
            duplicateDatasetContainer.click()
        }
    })
})

window.setupModal(btn_editdatasetmeta, datasetMetaContainer, () => {
    datasetMetaTitle.innerHTML = `Edit meta for: ${window.appState.currentDataset}`
    fixedFolderName = window.appState.currentDataset
    const datasetMeta = JSON.parse(fs.readFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/dataset_metadata.json`, "utf8"))
    const voiceId = datasetMeta.games[0].voiceId || fixedFolderName

    datasetMeta_voiceName.value = datasetMeta.games[0].voiceName
    datasetMeta_gameId.value = datasetMeta.games[0].gameId
    // datasetMeta_gameIdCode.value = datasetMeta.games[0].voiceId.includes("_") ? datasetMeta.games[0].voiceId.split("_")[0] : ""
    datasetMeta_gameIdCode.value = datasetMeta.games[0].voiceId.split("_")[0]
    datasetMeta_voiceId.value = datasetMeta.games[0].voiceId.split("_").slice(1, 10000).join("_")
    datasetMeta_langcode.value = datasetMeta.lang
    datasetMeta_modelVersion.value = parseFloat(datasetMeta.version.split(".").length==2 ? datasetMeta.version : datasetMeta.version.split(".").splice(0,2).join("."))
    datasetMeta_gender_male.checked = datasetMeta.games[0].gender=="male"
    datasetMeta_gender_female.checked = datasetMeta.games[0].gender=="female"
    datasetMeta_gender_other.checked = datasetMeta.games[0].gender=="other"
    composedVoiceId.innerHTML = voiceId
    datasetMeta_author.value = datasetMeta.author
    datasetMeta_license.value = datasetMeta.license || ""

    initDatasetMeta(() => {
        composedVoiceId.innerHTML = `${datasetMeta_gameIdCode.value}_${datasetMeta_voiceId.value}`
    })
    composedVoiceId.innerHTML = `${datasetMeta_gameIdCode.value}_${datasetMeta_voiceId.value}`
})

datasetMeta_gameIdCode.addEventListener("keyup", () => composedVoiceId.innerHTML = `${datasetMeta_gameIdCode.value}_${datasetMeta_voiceId.value}`)
datasetMeta_voiceId.addEventListener("keyup", () => composedVoiceId.innerHTML = `${datasetMeta_gameIdCode.value}_${datasetMeta_voiceId.value}`)


window.setupModal(btn_addmissingmeta, datasetMetaContainer, () => {
    datasetMetaTitle.innerHTML = `Add meta to: ${window.appState.currentDataset}`
    fixedFolderName = window.appState.currentDataset
    initDatasetMeta(() => {
        composedVoiceId.innerHTML = fixedFolderName
    })
    composedVoiceId.innerHTML = window.appState.currentDataset
    datasetMeta_voiceId.value = window.appState.currentDataset
})


window.setupModal(newDatasetButton, datasetMetaContainer, () => {
    datasetMetaTitle.innerHTML = "New dataset"
    fixedFolderName = undefined
    initDatasetMeta(() => {
        fs.mkdirSync(`${window.userSettings.datasetsPath}/${composedVoiceId.innerHTML}`)
        fs.mkdirSync(`${window.userSettings.datasetsPath}/${composedVoiceId.innerHTML}/wavs`)
        fs.writeFileSync(`${window.userSettings.datasetsPath}/${composedVoiceId.innerHTML}/metadata.csv`, [`000000.wav|Example Line|Example Line`].join("\n"))
    })
})
datasetMeta_voiceName.addEventListener("keyup", () => {
    if (voiceIDInputChanged) {
        return
    }

    if (datasetMeta_voiceName.value.trim().length) {
        datasetMeta_voiceId.value = datasetMeta_voiceName.value.trim().toLowerCase().replace(/\s/g, "_")
    }
    if (datasetMeta_voiceName.value.trim().length && datasetMeta_gameIdCode.value.trim().length) {
        composedVoiceId.innerHTML = `${datasetMeta_gameIdCode.value.trim().toLowerCase()}_${datasetMeta_voiceName.value.trim().toLowerCase().replace(/\s/g, "_")}`
    }
})

datasetMeta_voiceId.addEventListener("keyup", () => {
    voiceIDInputChanged = true
    if (datasetMeta_voiceId.value.trim().length && datasetMeta_gameIdCode.value.trim().length) {
        composedVoiceId.innerHTML = `${datasetMeta_gameIdCode.value}_${datasetMeta_voiceId.value.trim().toLowerCase()}`
    }
})
datasetMeta_gameIdCode.addEventListener("keyup", () => {
    if (fixedFolderName) {
        return
    }
    const voiceId = (voiceIDInputChanged ? datasetMeta_voiceId.value : datasetMeta_voiceName.value).trim().toLowerCase().replace(/\s/g, "_")
    if (voiceId.length && datasetMeta_gameIdCode.value.trim().length) {
        composedVoiceId.innerHTML = `${datasetMeta_gameIdCode.value.trim().toLowerCase()}_${voiceId}`
    }
})
datasetMeta_gender_male.addEventListener("change", () => {
    if (datasetMeta_gender_male.checked) {
        datasetMeta_gender_female.checked = false
        datasetMeta_gender_other.checked = false
    }
})
datasetMeta_gender_female.addEventListener("change", () => {
    if (datasetMeta_gender_female.checked) {
        datasetMeta_gender_male.checked = false
        datasetMeta_gender_other.checked = false
    }
})
datasetMeta_gender_other.addEventListener("change", () => {
    if (datasetMeta_gender_other.checked) {
        datasetMeta_gender_male.checked = false
        datasetMeta_gender_female.checked = false
    }
})
// =====================




ccLink.addEventListener("click", () => {
    shell.openExternal("https://creativecommons.org/licenses/")
})


// Settings
// ========
window.setupModal(settingsCog, settingsContainer)
// ========

// Tools
// ========
window.setupModal(toolsIcon, toolsContainer)
// ========

// Info
// ====
window.setupModal(infoIcon, infoContainer)
discordLink.addEventListener("click", () => {
    shell.openExternal("https://discord.gg/nv7c6E2TzV")
})
youtubeLink.addEventListener("click", () => {
    shell.openExternal("https://www.youtube.com/watch?v=PXv_SeTWk2M")
})
// ====

// Updates
// =======
window.setupModal(updatesIcon, updatesContainer)
const checkForUpdates = () => {
    doFetch("http://danruta.co.uk/xvatrainer_updates.txt").then(r=>r.json()).then(data => {
        fs.writeFileSync(`${path}/updates.json`, JSON.stringify(data), "utf8")
        checkUpdates.innerHTML = "Check for updates now"
        window.showUpdates()
    }).catch(() => {
        checkUpdates.innerHTML = "Can't reach server"
    })
}
window.showUpdates = () => {
    window.updatesLog = fs.readFileSync(`${path}/updates.json`, "utf8")
    window.updatesLog = JSON.parse(window.updatesLog)
    const sortedLogVersions = Object.keys(window.updatesLog).map( a => a.split('.').map( n => +n+100000 ).join('.') ).sort()
        .map( a => a.split('.').map( n => +n-100000 ).join('.') )

    const appVersion = window.appVersion.replace("v", "")
    const appIsUpToDate = sortedLogVersions.indexOf(appVersion)==(sortedLogVersions.length-1) || sortedLogVersions.indexOf(appVersion)==-1

    if (!appIsUpToDate) {
        update_nothing.style.display = "none"
        update_something.style.display = "block"
        updatesVersions.innerHTML = `This app version: ${appVersion}. Update available: ${sortedLogVersions[sortedLogVersions.length-1]}`
    } else {
        updatesVersions.innerHTML = `This app version: ${appVersion}. Up to date.`
    }

    updatesLogList.innerHTML = ""
    sortedLogVersions.reverse().forEach(version => {
        const versionLabel = createElem("h2", version)
        const logItem = createElem("div", versionLabel)
        window.updatesLog[version].split("\n").forEach(line => {
            logItem.appendChild(createElem("div", line))
        })
        updatesLogList.appendChild(logItem)
    })
}
checkForUpdates()
window.setupModal(updatesIcon, updatesContainer)

checkUpdates.addEventListener("click", () => {
    checkUpdates.innerHTML = "Checking for updates"
    checkForUpdates()
})
window.showUpdates()
// =======


// Patreon
// =======
window.setupModal(patreonIcon, patreonContainer, () => {
    const data = fs.readFileSync(`${path}/patreon.txt`, "utf8")
    const names = new Set()
    data.split("\r\n").forEach(name => names.add(name))

    let content = ``
    creditsList.innerHTML = ""
    names.forEach(name => content += `<br>${name}`)
    creditsList.innerHTML = content
})

// Training
// ========
window.setupModal(btn_trainmodel, trainContainer, () => {
    if (window.appState.currentDataset) {
        setTimeout(() => {
            const queueItem = window.training_state.datasetsQueue.filter(item => item.dataset_path.includes(window.appState.currentDataset))
            if (!queueItem.length) {
                window.showConfigMenu(`${window.userSettings.datasetsPath.replaceAll(/\\/, "/")}/${window.appState.currentDataset}`)
            }
        }, 500)
    }
})
window.setupModal(trainButton, trainContainer)
// ========