"use strict"

window.appVersion = "1.0"
app_version.innerHTML = "v"+window.appVersion
// const PRODUCTION = process.mainModule.filename.includes("resources")
const PRODUCTION = module.filename.includes("resources")
// const path = PRODUCTION ? `${__dirname.replace(/\\/g,"/")}/resources/app` : `${__dirname.replace(/\\/g,"/")}`
const path = PRODUCTION ? `${__dirname.replace(/\\/g,"/")}` : `${__dirname.replace(/\\/g,"/")}`
window.path = path
window.websocket_handlers = {}

const fs = require("fs")
const {exec} = require("child_process")
const {shell, ipcRenderer} = require("electron")
const {xVAAppLogger} = require("./javascript/appLogger.js")
window.appLogger = new xVAAppLogger(`./app.log`, window.appVersion)
window.appLogger.log(window.path)
require("./javascript/util.js")
require("./javascript/settingsMenu.js")
require("./javascript/tools.js")
require("./javascript/training.js")
const execFile = require('child_process').execFile
const spawn = require("child_process").spawn

// Start the server
if (window.PRODUCTION) {
    window.pythonProcess = spawn(`${path}/cpython_${window.userSettings.installation}/server.exe`, {stdio: "ignore"})
}
ipcRenderer.send('updateDiscord', {})

const initWebSocket = () => {

    window.ws = new WebSocket("ws://localhost:8000")

    ws.onopen = event => {
        console.log("WebSocket open")
        clearInterval(initWebSocketInterval)
    }

    ws.onclose = event => {
        console.log("WebSocket closed", event)
    }

    ws.onmessage = event => {
        const response = event.data ? JSON.parse(event.data) : undefined
        // console.log("response", response)

        if (Object.keys(window.websocket_handlers).includes(response["key"])) {
            window.websocket_handlers[response["key"]](response["data"])
        }
    }
}
const initWebSocketInterval = setInterval(initWebSocket, 1000)



window.datasets = {}
window.appState = {
    currentDataset: undefined,
    recordFocus: undefined
}
window.recordingState = {}
window.watchedModelsDirs = {}
setTimeout(() => {
    window.electron = require("electron")
}, 1000)



window.saveDatasetToFile = (datasetName) => {
    const metadata_out = []
    window.datasets[datasetName].metadata.forEach(row => {
        metadata_out.push(`${row[0].fileName}|${row[0].text}|${row[0].text}`)
    })
    fs.writeFileSync(`${window.userSettings.datasetsPath}/${datasetName}/metadata.csv`, metadata_out.join("\n"))
}


btn_addmissingmeta.addEventListener("click", () => {
    console.log("btn_addmissingmeta")
})

window.refreshDatasets = () => {
    // const datasetFolders = fs.readdirSync(`${window.path}/datasets`)
    const datasetFolders = fs.readdirSync(window.userSettings.datasetsPath).filter(name => !name.includes(".") && (fs.existsSync(`${window.userSettings.datasetsPath}/${name}/metadata.csv`) || fs.existsSync(`${window.userSettings.datasetsPath}/${name}/wavs`)))
    // const datasetFolders = fs.readdirSync(window.userSettings.datasetsPath).filter(name => !name.includes("."))
    const buttons = []
    datasetFolders.forEach(dataset => {
        window.datasets[dataset] = {metadata: []}

        const button = createElem("div.voiceType", dataset)

        button.addEventListener("click", event => {

            title.innerHTML = `Dataset: ${dataset}`

            const records = []
            const metadataAudioFiles = []
            window.appState.recordFocus = undefined

            if (fs.existsSync(`${window.userSettings.datasetsPath}/${dataset}/metadata.csv`)) {
                const data = fs.readFileSync(`${window.userSettings.datasetsPath}/${dataset}/metadata.csv`, "utf8")
                if (data.length) {
                    data.split("\n").forEach(line => {
                        metadataAudioFiles.push(line.split("|")[0])
                        records.push({
                            fileName: line.split("|")[0],
                            text: line.split("|")[1]
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

        })
        buttons.push(button)

        // if (!window.watchedModelsDirs.has(`${window.path}/datasets/${dataset}`)) {
        if (!Object.keys(window.watchedModelsDirs).includes(`${window.userSettings.datasetsPath}/${dataset}`)) {
            window.watchedModelsDirs[`${window.userSettings.datasetsPath}/${dataset}`] = fs.watch(`${window.userSettings.datasetsPath}/${dataset}`, {recursive: true, persistent: true}, (eventType, fileName) => {
                button.click()
            })
        }
        // window.watchedModelsDirs.add(`${window.path}/datasets/${dataset}`)

    })
    voiceTypeContainer.innerHTML = ""
    buttons.sort((a,b) => a.innerHTML<b.innerHTML?-1:1).forEach(button => voiceTypeContainer.appendChild(button))
}
window.refreshDatasets()






// Recording panel functionality
// =============================
window.setRecordFocus = ri => {
    if (window.appState.recordFocus!=undefined) {
        batchRecordsContainer.children[window.appState.recordFocus].style.background = "rgb(75,75,75)"
        batchRecordsContainer.children[window.appState.recordFocus].style.color = "white"
        Array.from(batchRecordsContainer.children[window.appState.recordFocus].children).forEach(c => {
            c.style.color = "white"
        })
    }

    if (ri!=undefined) {
        batchRecordsContainer.children[ri]
        batchRecordsContainer.children[ri].style.background = "rgb(255,255,255)"
        batchRecordsContainer.children[ri].style.color = "black"
        Array.from(batchRecordsContainer.children[ri].children).forEach(c => {
            c.style.color = "black"
        })

        textInput.value = window.datasets[window.appState.currentDataset].metadata[ri][0].text
        outFileNameInput.value = window.datasets[window.appState.currentDataset].metadata[ri][0].fileName
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
    setRecordFocus()
})
// =============================





// Button functionalities
// ======================
btn_save.addEventListener("click", () => {
    if (window.appState.recordFocus!=undefined) {

        const doTheRest = () => {
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

            window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][0].text = textInput.value.replace(/\|/g,"")
            window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][0].fileName = outFileNameInput.value.replace(".wav","")+".wav"
            window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][1].children[2].innerHTML = outFileNameInput.value.replace(".wav","")+".wav"
            window.datasets[window.appState.currentDataset].metadata[window.appState.recordFocus][1].children[3].innerHTML = textInput.value.replace(/\|/g,"")

            window.saveDatasetToFile(window.appState.currentDataset)
            setRecordFocus()
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
        const recordFocus = window.appState.recordFocus
        const text = window.datasets[window.appState.currentDataset].metadata[recordFocus][0].text
        confirmModal(`Are you sure you'd like to delete this line?<br><br><i>${text}</i>`).then(confirmation => {
            if (confirmation) {
                const fileName = window.datasets[window.appState.currentDataset].metadata[recordFocus][0].fileName
                delete window.datasets[window.appState.currentDataset].metadata[recordFocus]
                window.saveDatasetToFile(window.appState.currentDataset)

                if (fs.existsSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${fileName}`)) {
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
                    createModal("spinner", "Auto-transcribing...<br>This may take a few minutes if there are hundreds of lines.<br>A model will automatically be downloaded the first ever time this is done, and may then take a little longer.<br>Audio files must be mono 22050Hz<br><br>This window will close if there is an error.")
                    window.autotranscribe(window.appState.currentDataset)
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







window.autotranscribe = dataset => {

    const metadataFilePath = `${window.userSettings.datasetsPath}/${dataset}/metadata.csv`

    execFile(`${window.path}/autotranscribe.exe`, [metadataFilePath], (err, stdout, stderr) => {
        if (err && stderr) {
            window.appLogger.log(`Error auto-transcribing file: ${metadataFilePath}`)
            window.appLogger.log(stderr)
        }
        closeModal()
    })
}
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
            console.log("records", records)
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
            console.log("cleanedData", cleanedData)
            populateRecordsList(window.appState.currentDataset, cleanedData, true)
            refreshRecordsList(window.appState.currentDataset)
            saveDatasetToFile(window.appState.currentDataset)
        } else {
            // batch_clearBtn.click()
        }

        batchDropZoneNote.innerHTML = "Drag and drop .txt or .csv files here"
    }
}
window.refreshRecordsList = (dataset) => {
    if (window.tools_state.running) {
        return
    }
    batchRecordsContainer.innerHTML = ""
    window.datasets[dataset].metadata.forEach(recordAndElem => {
        recordAndElem[1].children[0].innerHTML = batchRecordsContainer.children.length.toString()
        batchRecordsContainer.appendChild(recordAndElem[1])
    })
}


const populateRecordsList = (dataset, records, additive=false) => {

    if (!additive) {
        window.datasets[dataset].metadata = []
    }

    batchDropZoneNote.style.display = "none"

    records.forEach((record, ri) => {
        const row = createElem("div.row")

        const rNumElem = createElem("div.rowItem", batchRecordsContainer.children.length.toString())
        const rGameElem = createElem("div.rowItem", "")



        record.fileName = record.fileName==undefined ? getNextFileName(dataset)+".wav" : record.fileName
        const rTextElem = createElem("div.rowItem", record.text)
        const rOutPathElem = createElem("div.rowItem", "&lrm;"+record.fileName+"&lrm;")


        row.appendChild(rNumElem)
        const potentialAudioPath = `${window.userSettings.datasetsPath}/${window.appState.currentDataset}/wavs/${record.fileName}`
        let observer
        if (fs.existsSync(potentialAudioPath)) {
            // const audio = createElem("audio", {controls: true}, createElem("source", {
            //     src: potentialAudioPath,
            //     type: `audio/wav`
            // }))
            // audio.style.zIndex = "1"
            // rGameElem.style.zIndex = "1"
            // rGameElem.appendChild(audio)

            const anchorSpan = createElem("span")
            rGameElem.appendChild(anchorSpan)

            observer = new IntersectionObserver((entry, observer) => {
                // console.log("entry", entry, record.text, entry[0].isIntersecting)
                if (entry[0].isIntersecting) {
                    if (window.datasets[dataset].metadata[ri][1].children[1].children.length<2) {
                        // console.log("here", ri)
                        const audio = createElem("audio", {controls: true}, createElem("source", {
                            src: potentialAudioPath,
                            type: `audio/wav`
                        }))
                        audio.style.zIndex = "1"
                        // console.log("window.datasets[dataset].metadata[ri][1].children[1]", )
                        window.datasets[dataset].metadata[ri][1].children[1]
                        window.datasets[dataset].metadata[ri][1].children[1].appendChild(audio)
                    }

                    observer.unobserve(row)
                    observer.disconnect()
                }

            // }, {root: batchRecordsContainer, rootMargin: "0px 0px -100px 0px", threshold: 1.0})
            // }, {rootMargin: "0px 0px -100px 0px", threshold: 1.0})
            }, {threshold: 0})
            // }, {root: null})
            observer.observe(row)

        }
        row.appendChild(rGameElem)

        row.appendChild(rOutPathElem)
        row.appendChild(rTextElem)

        row.addEventListener("click", event => {
            setRecordFocus(ri)
        })

        window.datasets[dataset].metadata.push([record, row, ri, observer])
    })
}

batch_main.addEventListener("dragenter", event => uploadBatchCSVs("dragenter", event), false)
batch_main.addEventListener("dragleave", event => uploadBatchCSVs("dragleave", event), false)
batch_main.addEventListener("dragover", event => uploadBatchCSVs("dragover", event), false)
batch_main.addEventListener("drop", event => uploadBatchCSVs("drop", event), false)
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

                const outputNameSox = `${window.recordingState.tempAudioFileName.replace(".wav", "")}_sil.wav`
                const soxCommand = `sox "${window.recordingState.tempAudioFileName}" "${outputNameSox}" noisered ${window.path}/noise_profile_file ${window.userSettings.noiseRemStrength} rate 22050`
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

        fs.writeFileSync(`${window.userSettings.datasetsPath}/${composedVoiceId.innerHTML}/dataset_metadata.json`, JSON.stringify({
            "version": "2.0",
            "modelVersion": parseFloat(datasetMeta_modelVersion.value).toFixed(1),
            "modelType": "FastPitch",
            "emb_size": 2,
            "games": [
                {
                    "gameId": datasetMeta_gameId.value.trim().toLowerCase(),
                    "voiceId": datasetMeta_voiceId.value.trim().toLowerCase(),
                    "voiceName": datasetMeta_voiceName.value.trim(),
                    "gender": (datasetMeta_gender_male.checked ? "male" : (datasetMeta_gender_female.checked ? "female" : "other")),
                    "lang": datasetMeta_langcode.value.trim().toLowerCase(),
                    "emb_i": 0
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

window.setupModal(btn_editdatasetmeta, datasetMetaContainer, () => {
    datasetMetaTitle.innerHTML = `Edit meta for: ${window.appState.currentDataset}`
    fixedFolderName = window.appState.currentDataset
    const datasetMeta = JSON.parse(fs.readFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/dataset_metadata.json`, "utf8"))
    console.log("datasetMeta", datasetMeta)

    datasetMeta_voiceName.value = datasetMeta.games[0].voiceName
    datasetMeta_gameId.value = datasetMeta.games[0].gameId
    datasetMeta_gameIdCode.value = datasetMeta.games[0].voiceId.includes("_") ? datasetMeta.games[0].voiceId.split("_")[0] : ""
    datasetMeta_voiceId.value = datasetMeta.games[0].voiceId
    datasetMeta_langcode.value = datasetMeta.games[0].lang
    datasetMeta_modelVersion.value = parseFloat(datasetMeta.version.split(".").length==2 ? datasetMeta.version : datasetMeta.version.split(".").splice(0,2).join("."))
    datasetMeta_gender_male.checked = datasetMeta.games[0].gender=="male"
    datasetMeta_gender_female.checked = datasetMeta.games[0].gender=="female"
    datasetMeta_gender_other.checked = datasetMeta.games[0].gender=="other"
    composedVoiceId.innerHTML = fixedFolderName

    initDatasetMeta(() => {
        composedVoiceId.innerHTML = fixedFolderName
    })
    composedVoiceId.innerHTML = window.appState.currentDataset
    datasetMeta_voiceId.value = window.appState.currentDataset
})

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
    if (fixedFolderName) {
        return
    }
    voiceIDInputChanged = true
    if (datasetMeta_voiceId.value.trim().length && datasetMeta_gameIdCode.value.trim().length) {
        composedVoiceId.innerHTML = `${datasetMeta_gameIdCode.value.trim().toLowerCase()}_${datasetMeta_voiceId.value.trim().toLowerCase().replace(/\s/g, "_")}`
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



// newDatasetButton.addEventListener("click", () => {
//     window.createModal("prompt", {prompt: "Name your dataset", value: cachedName}).then(name => {
//         if (!name.length) {
//             return
//         }

//         if (Object.keys(window.datasets).includes(name)) {
//             setTimeout(() => {
//                 window.createModal("error", "This dataset name already exists").then(() => {
//                     setTimeout(() => {
//                         cachedName = name
//                         newDatasetButton.click()
//                     }, 500)
//                 })
//             }, 500)
//             return

//         }

//         cachedName = ""

//         console.log(name)

//         fs.mkdirSync(`${window.path}/datasets/${name}`)
//         window.refreshDatasets()
//     })
// })











// Settings
// ========
window.setupModal(settingsCog, settingsContainer)
// ========

// Tools
// ========
window.setupModal(toolsIcon, toolsContainer)
// ========