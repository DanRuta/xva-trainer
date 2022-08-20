"use strict"

const nodeDiskInfo = require("node-disk-info")
try {
    const disks = nodeDiskInfo.getDiskInfoSync()
    for (const disk of disks) {
        disk_drive_select.appendChild(createElem(`option#drive_${disk._mounted.split(":")[0].toLowerCase()}`, disk._mounted))
    }
} catch (e) {
    console.log(e)
}

// https://www.chartjs.org/docs/latest/
window.updateTrainingGraphValues = (allValues, chart) => {
    chart.data.datasets[0].data = allValues.map(pairs => pairs[1])
    chart.data.labels = allValues.map(pairs => pairs[0])
    chart.update()
}


window.initChart = (canvasId, title, colour, {boundsMax=100, startingData=undefined, max=60, minY=0, extradataset=undefined}={}) => {

    const datasets = [
        {
            label: title,
            data: Array.from(Array(60)).map((_,i)=>0),
            borderColor: `rgb(${colour})`,
            backgroundColor: `rgba(${colour},0.1)`,
        }
    ]
    if (extradataset) {
        datasets.push(extradataset)
    }

    const ctx = document.getElementById(canvasId).getContext('2d')
    const chart_object = new Chart(ctx, {
        type: 'line',
        data: {
            labels: startingData?startingData:Array.from(Array(60)).map((_,i)=>i),
            datasets: datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            elements: {
                point: {
                    radius: 0
                },
                line: {
                    fill: true,
                    borderWidth: 2
                }
            },
            scales: {
                x: {
                    display: false,
                    min: 0,
                    max: max,
                },
                y: {
                    display: true,
                    beginAtZero: false,
                    suggestedMin: minY,
                    suggestedMax: boundsMax
                }
            }
        }
    })
    return chart_object
}

const startingData = []
window.loss_chart_object = window.initChart("loss_plot", "Loss", "255,112,67", {startingData: startingData, max: 100000, minY: null, boundsMax: null})
window.loss_delta_chart_object = window.initChart("loss_delta_plot", "Loss Delta", "255,112,67", {startingData: startingData, max: 100000, boundsMax: null, extradataset: {
        label: "Target",
        borderColor: `rgb(50,50,250)`,
        data: Array.from(Array(60)).map((_,i)=>0),
    }})

window.cpu_chart_object = initChart("cpu_chart", "CPU", "62,149,205")
window.cuda_chart_object = initChart("cuda_chart", "CUDA", "62,149,205")
window.ram_chart_object = undefined
window.vram_chart_object = undefined
window.disk_chart_object = initChart("disk_chart", "Disk", "140,200,130")



// https://stackoverflow.com/questions/52822588/how-to-get-calculate-the-current-disk-activity-in-node-js-on-windows
let diskTrackerProcess
let diskTrackerCurrent = 0
window.getDiskTimePercent = (driveLetter) => {

    if (!diskTrackerProcess) {
        diskTrackerProcess = exec(`typeperf.exe "\\LogicalDisk(${driveLetter})\\% Disk Time" -SC 50`);
        diskTrackerProcess.stdout.on('data', function (data) {
            const stdout = data.toString();
            const lines = stdout.split('\r\n');
            const values = lines[0].split(",");
            const result = Math.round(values.pop().replace(/"/g, ''), 0);
            if (!isNaN(result)) {
                diskTrackerCurrent = result;
            }
        });
        diskTrackerProcess.on('exit', function (code) {
            diskTrackerProcess = null;
        });
        return diskTrackerCurrent;
    } else {
        return diskTrackerCurrent;
    }
}
window.updateSystemGraphs = () => {

    const totalRam = window.totalmem()
    const ramUsage = window.totalmem() - window.freemem()
    const percentRAM = ramUsage/totalRam*100

    if (!window.ram_chart_object) {
        window.ram_chart_object = initChart("ram_chart", "RAM", "191,101,191", {boundsMax: totalRam/1024})
    }

    window.cpuUsage(cpuLoad => {
        window.smi((err, data) => {
            let totalVRAM
            let usedVRAM
            let computeLoad
            let percent
            let vramUsage

            if (window.userSettings.installation=="gpu" && !err) {
                if (data.nvidia_smi_log.gpu.length) {
                    totalVRAM = parseInt(data.nvidia_smi_log.gpu[0].fb_memory_usage.total.split(" ")[0])
                    usedVRAM = parseInt(data.nvidia_smi_log.gpu[0].fb_memory_usage.used.split(" ")[0])
                    computeLoad = parseFloat(data.nvidia_smi_log.gpu[0].utilization.gpu_util.split(" ")[0])
                } else {
                    totalVRAM = parseInt(data.nvidia_smi_log.gpu.fb_memory_usage.total.split(" ")[0])
                    usedVRAM = parseInt(data.nvidia_smi_log.gpu.fb_memory_usage.used.split(" ")[0])
                    computeLoad = parseFloat(data.nvidia_smi_log.gpu.utilization.gpu_util.split(" ")[0])
                }
            }


            if (!window.vram_chart_object) {
                window.vram_chart_object = initChart("vram_chart", "VRAM", "62,149,205", {boundsMax: parseInt(totalVRAM/1024)})
            }

            if (totalVRAM) {
                percent = usedVRAM/totalVRAM*100
                vramUsage = `${(usedVRAM/1000).toFixed(1)}/${(totalVRAM/1000).toFixed(1)} GB (${percent.toFixed(2)}%)`
            }

            // CPU
            cpu_chart_object.data.datasets[0].data.push(parseInt(cpuLoad*100))
            if (cpu_chart_object.data.datasets[0].data.length>60) {
                cpu_chart_object.data.datasets[0].data.splice(0,1)
            }
            cpu_chart_object.data.datasets[0].label = `${cpu_chart_object.data.datasets[0].label.split(" ")[0]} ${(cpuLoad*100).toFixed(2)}%`

            // RAM
            ram_chart_object.data.datasets[0].data.push(parseInt(ramUsage/1024))
            if (ram_chart_object.data.datasets[0].data.length>60) {
                ram_chart_object.data.datasets[0].data.splice(0,1)
            }
            ram_chart_object.data.datasets[0].label = `${ram_chart_object.data.datasets[0].label.split(" ")[0]} ${(ramUsage/1024).toFixed(1)}GB/${(totalRam/1024).toFixed(1)}GB`

            if (computeLoad) {
                // CUDA
                cuda_chart_object.data.datasets[0].data.push(parseInt(computeLoad))
                if (cuda_chart_object.data.datasets[0].data.length>60) {
                    cuda_chart_object.data.datasets[0].data.splice(0,1)
                }
                cuda_chart_object.data.datasets[0].label = `${cuda_chart_object.data.datasets[0].label.split(" ")[0]} ${computeLoad}%`

                // VRAM
                vram_chart_object.data.datasets[0].data.push(parseInt(usedVRAM/1024))
                if (vram_chart_object.data.datasets[0].data.length>60) {
                    vram_chart_object.data.datasets[0].data.splice(0,1)
                }
                vram_chart_object.data.datasets[0].label = `${vram_chart_object.data.datasets[0].label.split(" ")[0]} ${(usedVRAM/1024).toFixed(1)}/${(totalVRAM/1024).toFixed(1)} GB (${percent.toFixed(2)}%)`
            }


            // Disk
            try {
                const diskUsage = getDiskTimePercent(disk_drive_select.selectedOptions[0].innerText)
                disk_chart_object.data.datasets[0].data.push(parseInt(diskUsage))
                if (disk_chart_object.data.datasets[0].data.length>60) {
                    disk_chart_object.data.datasets[0].data.splice(0,1)
                }
                disk_chart_object.data.datasets[0].label = `${disk_chart_object.data.datasets[0].label.split(" ")[0]} (${disk_drive_select.selectedOptions[0].innerText}) ${(diskUsage).toFixed(2)}%`
            } catch (e) {

            }


            cpu_chart_object.update()
            ram_chart_object.update()
            if (vram_chart_object) {
                vram_chart_object.update()
            }
            if (cuda_chart_object) {
                cuda_chart_object.update()
            }
            disk_chart_object.update()
            window.updateSystemGraphs()


        })
    })
}
window.updateSystemGraphs()
disk_drive_select.addEventListener("change", () => {
    for (let x=0; x<60; x++) {
        disk_chart_object.data.datasets[0].data.push(0)
        if (disk_chart_object.data.datasets[0].data.length>60) {
            disk_chart_object.data.datasets[0].data.splice(0,1)
        }
        disk_chart_object.update()
    }
    disk_chart_object.data.datasets[0].label = `${disk_chart_object.data.datasets[0].label.split(" ")[0]} (${disk_drive_select.selectedOptions[0].innerText}) 0%`
    disk_chart_object.update()
})


window.training_state = {
    // state: 0, // 0 idle, 1 training, 2 paused, 3 finished
    isBatchTraining: false,
    stage_viewed: 1,
    dirs_watching: [],
    datasetsQueue: [
    ],
    trainingQueueItem: 0,
    websocketDynamicLogLine: "",
    currentlyViewedDataset: undefined,
    selectedQueueItem: undefined,

}



window.saveTrainingQueueJSON = () => {
    // Copy over everything except the DOM element references
    const saveJSON = window.training_state.datasetsQueue.map(dataset => {
        const filteredJSON = {}
        filteredJSON.voice_name = dataset.voice_name
        filteredJSON.dataset_path = dataset.dataset_path
        filteredJSON.output_path = dataset.output_path
        filteredJSON.batch_size = dataset.batch_size
        filteredJSON.num_workers = dataset.num_workers
        filteredJSON.epochs_per_checkpoint = dataset.epochs_per_checkpoint
        filteredJSON.force_stage = dataset.force_stage
        filteredJSON.status = dataset.status
        filteredJSON.hifigan_checkpoint = dataset.hifigan_checkpoint
        return filteredJSON
    })

    fs.writeFileSync(`${window.path}/training_queue.json`, JSON.stringify(window.training_state.datasetsQueue, null, 4))
}

window.refreshTrainingQueueList = () => {

    trainingQueueRecordsContainer.innerHTML = ""

    window.training_state.datasetsQueue.forEach((dataset, di) => {

        const rowElem = createElem("div")
        const queueItem = createElem("div", `${di}`)
        const configButton = createElem("button.smallButton", "Config")
        const buttonContainer = createElem("div")
        buttonContainer.appendChild(configButton)
        const statusBox = createElem("div", dataset.status)
        if (dataset.status=="Finished") {
            statusBox.style.background = "green"
            statusBox.style.color = "white"
        }

        const nameBox = createElem("div", dataset.dataset_path.split("/").reverse()[0])
        nameBox.title = dataset.dataset_path.split("/").reverse()[0]

        rowElem.appendChild(queueItem)
        rowElem.appendChild(buttonContainer)
        rowElem.appendChild(statusBox)
        rowElem.appendChild(nameBox)
        trainingQueueRecordsContainer.appendChild(rowElem)

        window.training_state.datasetsQueue[di].rowElem = rowElem

        rowElem.addEventListener("click", event => {
            if (event.target==configButton) {
                return
            }

            trainingQueueBtnMoveUp.disabled = false
            trainingQueueBtnMoveDown.disabled = false
            trainingQueueBtnDelete.disabled = false
            trainingOpenCkpts.disabled = false

            if (window.training_state.selectedQueueItem!=di) {

                // Unhighlight previous
                const allRowElems = Array.from(document.querySelectorAll("#trainingQueueRecordsContainer>div"))
                if (window.training_state.selectedQueueItem!=undefined) {
                    allRowElems[window.training_state.selectedQueueItem].style.background = "none"
                    Array.from(allRowElems[window.training_state.selectedQueueItem].children).forEach(child => child.style.color = "white")
                }
                window.training_state.selectedQueueItem = di

                // Highlight this one
                allRowElems[window.training_state.selectedQueueItem].style.background = "white"
                Array.from(allRowElems[window.training_state.selectedQueueItem].children).forEach(child => child.style.color = "black")


                // View this dataset in the text/graph logs
                window.training_state.currentlyViewedDataset = dataset.dataset_path
                window.training_state.trainingQueueItem = di
                trainingStartBtn.disabled = false
                training_title.innerHTML = `Training - ${nameBox.title}`

                if (!fs.existsSync(dataset.output_path)) {
                    fs.mkdirSync(dataset.output_path)
                }

                console.log(`${dataset.output_path}/${dataset.dataset_path.split("/").reverse()[0]}`)
                startTrackingFolder(dataset.dataset_path, `${dataset.output_path}/${dataset.dataset_path.split("/").reverse()[0]}`)
                window.updateTrainingLogText()
                window.updateTrainingGraphs()

                cmdPanel.scrollTop = cmdPanel.scrollHeight
            }
            window.training_state.selectedQueueItem = di
        })
        if (window.training_state.selectedQueueItem==di) {
            rowElem.style.background = "white"
            Array.from(rowElem.children).forEach(child => child.style.color = "black")
        }

        configButton.addEventListener("click", () => {
            configAnExistingItem = true
            window.training_state.datasetsQueue[di].dataset_path = window.training_state.datasetsQueue[di].dataset_path.replaceAll(/\\/, "/")
            window.training_state.datasetsQueue[di].output_path = window.training_state.datasetsQueue[di].output_path.replaceAll(/\\/, "/")
            window.showConfigMenu(window.training_state.datasetsQueue[di], di)
        })
    })
}


// Load training batch queue file
try {
    if (fs.existsSync(`${window.path}/training_queue.json`)) {
        fs.readFile(`${window.path}/training_queue.json`, "utf8", (err, data) => {
            if (data) {
                window.training_state.datasetsQueue = JSON.parse(data)
                window.refreshTrainingQueueList()
                window.saveTrainingQueueJSON()
            }
        })
    } else {
        window.training_state.datasetsQueue = []
        window.refreshTrainingQueueList()
        window.saveTrainingQueueJSON()
        console.log("new file")
    }
} catch (e) {
    console.log(e)
}


window.training_state.trainingQueueItem = 0
window.training_state.stage_viewed = 1

window.trainingTextLogNumLines = 0
window.updateTrainingLogText = () => {
    try {
        const datasetConfig = window.training_state.datasetsQueue.find(ds => ds.dataset_path==window.training_state.currentlyViewedDataset)
        const datasetId = datasetConfig.dataset_path.split("/").reverse()[0]

        if (fs.existsSync(`${datasetConfig.output_path}/${datasetId}/training.log`)) {
            const logData = fs.readFileSync(`${datasetConfig.output_path}/${datasetId}/training.log`, "utf8").split("\n")
            logData.push(window.training_state.websocketDynamicLogLine)
            cmdPanel.innerHTML = logData.join("<br>")+"<br><br>"
            // TODO, only apply if the number of lines changes
            if (window.trainingTextLogNumLines!=logData.length) {
                cmdPanel.scrollTop = cmdPanel.scrollHeight
                window.trainingTextLogNumLines = logData.length
            }
        }
    } catch (e) {
        console.log(e)
    }
}
window.updateTrainingGraphs = () => {

    const datasetConfig = window.training_state.datasetsQueue.find(ds => ds.dataset_path==window.training_state.currentlyViewedDataset)
    const datasetId = datasetConfig.dataset_path.split("/").reverse()[0]
    if (fs.existsSync(`${datasetConfig.output_path}/${datasetId}/graphs.json`)) {
        const jsonString = fs.readFileSync(`${datasetConfig.output_path}/${datasetId}/graphs.json`, "utf8")

        try {
            const graphsJson = JSON.parse(jsonString)
            const stageData = graphsJson.stages[window.training_state.stage_viewed]

            if (window.graphs_only_latest) {
                stageData.loss = stageData.loss.slice(stageData.loss.length-window.graphs_only_latest, stageData.loss.length)
                stageData.loss_delta = stageData.loss_delta.slice(stageData.loss_delta.length-window.graphs_only_latest, stageData.loss_delta.length)
            }

            window.updateTrainingGraphValues(stageData.loss, window.loss_chart_object)
            window.updateTrainingGraphValues(stageData.loss_delta, window.loss_delta_chart_object)


            // window.loss_delta_chart_object.data.datasets[1].data = stageData.loss_delta.map((_,i)=>0.04)
            if (stageData.target_delta) {
                window.loss_delta_chart_object.data.datasets[1].data = stageData.loss_delta.map((_,i)=>stageData.target_delta)
                window.loss_delta_chart_object.data.labels = stageData.loss_delta.map(pairs => pairs[0])
                window.loss_delta_chart_object.update()
            }


        } catch (e) {
            console.log(e)
            // console.log(`${datasetConfig.output_path}/graphs.json`)
        }
    }
}
graphs_only_latest_chbx.addEventListener("click", () => {
    window.graphs_only_latest = graphs_only_latest_chbx.checked ? parseInt(graphs_only_latest_val.value) : undefined
    localStorage.setItem("graphs_only_latest", window.graphs_only_latest)
    window.updateTrainingGraphs()
})
graphs_only_latest_val.addEventListener("keyup", () => {
    window.graphs_only_latest = graphs_only_latest_chbx.checked ? parseInt(graphs_only_latest_val.value) : undefined
    localStorage.setItem("graphs_only_latest", window.graphs_only_latest)
    window.updateTrainingGraphs()
})
graphs_only_latest_val.addEventListener("change", () => {
    window.graphs_only_latest = graphs_only_latest_chbx.checked ? parseInt(graphs_only_latest_val.value) : undefined
    localStorage.setItem("graphs_only_latest", window.graphs_only_latest)
    window.updateTrainingGraphs()
})

window.graphs_only_latest = localStorage.getItem("graphs_only_latest")
if (window.graphs_only_latest) {
    graphs_only_latest_chbx.checked = true
    graphs_only_latest_val.value = parseInt(window.graphs_only_latest)
    window.graphs_only_latest = parseInt(window.graphs_only_latest)
}



const startTrackingFolder = (dataset_path, output_path) => {
    if (!Object.keys(window.training_state.dirs_watching).includes(output_path)) {
        if (!fs.existsSync(output_path)) {
            fs.mkdirSync(output_path)
        }
        const folderWatch = fs.watch(output_path, {recursive: false, persistent: true}, (eventType, fileName) => {

            if (window.training_state.currentlyViewedDataset==dataset_path && fileName=="training.log") {
                window.updateTrainingLogText()
            }

            if (window.training_state.currentlyViewedDataset==dataset_path && fileName=="graphs.json") {
                window.updateTrainingGraphs()
            }
        })
        window.training_state.dirs_watching[output_path] = folderWatch
    }
}


window.nextTrainingItem = () => {
    window.training_state.trainingQueueItem++
}


window.stopTraining = () => {
    window.training_state.dirs_watching.forEach(folder => folder.close())
}


trainingQueueBtnMoveUp.addEventListener("click", () => {
    if (window.training_state.selectedQueueItem==undefined || window.training_state.selectedQueueItem==0) {
        return
    }

    const selectedItem = window.training_state.datasetsQueue.splice(window.training_state.selectedQueueItem, 1)[0]
    window.training_state.datasetsQueue.splice(window.training_state.selectedQueueItem-1, 0, selectedItem)
    window.training_state.selectedQueueItem-=1

    window.refreshTrainingQueueList()
    window.saveTrainingQueueJSON()
})


trainingQueueBtnMoveDown.addEventListener("click", () => {
    if (window.training_state.selectedQueueItem==undefined || window.training_state.selectedQueueItem==(window.training_state.datasetsQueue.length-1)) {
        return
    }

    const selectedItem = window.training_state.datasetsQueue.splice(window.training_state.selectedQueueItem, 1)[0]
    window.training_state.datasetsQueue.splice(window.training_state.selectedQueueItem+1, 0, selectedItem)
    window.training_state.selectedQueueItem+=1

    window.refreshTrainingQueueList()
    window.saveTrainingQueueJSON()
})

trainingQueueBtnDelete.addEventListener("click", () => {
    if (window.training_state.selectedQueueItem==undefined) {
        return
    }
    window.confirmModal("Are you sure you'd like to remove this dataset from the training queue?").then(response => {
        if (response) {
            window.training_state.datasetsQueue.splice(window.training_state.selectedQueueItem, 1)[0]
            window.training_state.selectedQueueItem = undefined
            window.refreshTrainingQueueList()
            window.saveTrainingQueueJSON()
        }
    })
})

trainingQueueBtnBatchTrain.addEventListener("click", () => {
    if (window.userSettings.installation != "gpu") {
        window.errorModal("The CPU-only installation is enabled. Switch to the GPU installation if you can, or run training on a CUDA-compatible machine if not. It is not practical to run the training on the CPU, it would take forever.")
        return
    }

    if (!window.training_state.datasetsQueue.length) {
        return window.errorModal("Add some datasets to the queue first, either via the Add button, or via the Train button from the dataset view in the main app")
    }

    // Go down the list, until the first item is found that is not yet finished
    let itemIndex = 0
    do {
        if (window.training_state.datasetsQueue[itemIndex].status!="Finished") {
            break
        }
        itemIndex++

    } while (itemIndex<window.training_state.datasetsQueue.length)

    if (itemIndex>=window.training_state.datasetsQueue.length) {
        return window.errorModal("All datasets have finished training. Add some more!")
    }


    window.confirmModal("Are you sure you'd like to start batch training your datasets?").then(response => {
        if (response) {
            window.training_state.isBatchTraining = true
            window.training_state.trainingQueueItem = itemIndex
            const allRowElems = Array.from(document.querySelectorAll("#trainingQueueRecordsContainer>div"))
            allRowElems[itemIndex].click()
            trainingStartBtn.click()
        }
    })
})






stage_select.addEventListener("change", () => {
    window.training_state.stage_viewed = parseInt(stage_select.value.replace("stage", ""))
    window.updateTrainingLogText()
    window.updateTrainingGraphs()
})




fp_ckpt_option_other.addEventListener("click", () => trainingAddConfigCkptPathInput.disabled = !fp_ckpt_option_other.checked)
fp_ckpt_option_other.addEventListener("change", () => trainingAddConfigCkptPathInput.disabled = !fp_ckpt_option_other.checked)
fp_ckpt_option_female.addEventListener("change", () => trainingAddConfigCkptPathInput.disabled = !fp_ckpt_option_other.checked)
fp_ckpt_option_male.addEventListener("change", () => trainingAddConfigCkptPathInput.disabled = !fp_ckpt_option_other.checked)

window.showConfigMenu = (startingData, di) => {

    queueItemConfigModalContainer.style.opacity = 1
    queueItemConfigModalContainer.style.display = "flex"

    const configData = (typeof startingData == "string" ? undefined : startingData) || {
        "status": "Ready",

        "dataset_path": undefined,
        "output_path": undefined,
        "checkpoint": undefined,
        "hifigan_checkpoint": undefined,

        // "use_amp": "true",
        // "num_workers": 4, // TODO, app-level default settings
        // "batch_size": 8, // TODO, app-level default settings
        // "epochs_per_checkpoint": 3, // TODO, app-level default settings
        "force_stage": undefined,
    }
    if (typeof startingData == "string") {
        configData["dataset_path"] = startingData
    }

    window.training_state.currentlyConfiguringDatasetI = di

    trainingAddConfigDatasetPathInput.value = window.userSettings.datasetsPath
    trainingAddConfigDatasetPathInput.disabled = !!startingData

    trainingAddConfigDoForceStageCkbx.checked = configData.force_stage!=undefined
    if (configData.force_stage!=undefined) {
        trainingAddConfigForceStageNumberSelect.value = configData.force_stage
        trainingAddConfigForceStageNumberSelect.disabled = false
    } else {
        trainingAddConfigForceStageNumberSelect.value = "1"
        trainingAddConfigForceStageNumberSelect.disabled = true
    }

    trainingAddConfigDatasetPathInput.value = configData.dataset_path || window.userSettings.datasetsPath
    trainingAddConfigOutputPathInput.value = configData.output_path || ""
    fp_ckpt_option_male.checked = configData.checkpoint=="[male]"
    fp_ckpt_option_female.checked = configData.checkpoint=="[female]"
    fp_ckpt_option_other.checked = configData.checkpoint!="[female]" && configData.checkpoint!="[male]"
    trainingAddConfigCkptPathInput.value = configData.checkpoint || ""
    // trainingAddConfigHiFiCkptPathInput.value = configData.hifigan_checkpoint || ""

    hifigan_ckpt_option_male.checked = configData.hifigan_checkpoint=="[male]"
    hifigan_ckpt_option_female.checked = configData.hifigan_checkpoint=="[female]"

    if (configData.num_workers !== undefined) {
        trainingAddConfigWorkersInput.value = parseInt(configData.num_workers)
    } else {
        trainingAddConfigBatchSizeInput.value = window.localStorage.getItem("training.batch_size")||"8"
    }
    if (configData.batch_size !== undefined) {
        trainingAddConfigBatchSizeInput.value = parseInt(configData.batch_size)
    } else {
        trainingAddConfigEpochsPerCkptInput.value = window.localStorage.getItem("training.epochs_per_ckpt")||"3"
    }
    if (configData.epochs_per_checkpoint !== undefined) {
        trainingAddConfigEpochsPerCkptInput.value = parseInt(configData.epochs_per_checkpoint)
    } else {
        trainingAddConfigUseAmp.checked = !!parseInt(window.localStorage.getItem("training.useFP16"))
    }
    if (configData.use_amp !== undefined) {
        trainingAddConfigUseAmp.checked = configData.use_amp ? configData.use_amp=="true" : true
    } else {
        trainingAddConfigWorkersInput.value = window.localStorage.getItem("training.num_workers")||"3"
    }

    queueItemConfigModalContainer.style.display = "flex"
}

trainingAddConfigDoForceStageCkbx.addEventListener("change", () => {
    trainingAddConfigForceStageNumberSelect.disabled = !trainingAddConfigDoForceStageCkbx.checked
})


let configAnExistingItem = false
cancelConfig.addEventListener("click", () => {
    queueItemConfigModalContainer.style.display = "none"
})
trainingAddConfigDatasetPathInput.addEventListener("keyup", e => {
    trainingAddConfigDatasetPathInput.value = trainingAddConfigDatasetPathInput.value.replaceAll(/\\/, "/")
})
trainingAddConfigOutputPathInput.addEventListener("keyup", e => {
    trainingAddConfigOutputPathInput.value = trainingAddConfigOutputPathInput.value.replaceAll(/\\/, "/")
})
acceptConfig.addEventListener("click", () => {

    trainingAddConfigDatasetPathInput.value = trainingAddConfigDatasetPathInput.value.replaceAll(/\\/, "/")
    trainingAddConfigOutputPathInput.value = trainingAddConfigOutputPathInput.value.replaceAll(/\\/, "/")

    if (!trainingAddConfigDatasetPathInput.value.trim().length) {
        return window.errorModal("You need to specify where your dataset is located.", queueItemConfigModalContainer)
    }
    if (!trainingAddConfigOutputPathInput.value.trim().length) {
        return window.errorModal("You need to specify where to output the intermediate data/models for your dataset.", queueItemConfigModalContainer)
    }
    if (!fp_ckpt_option_other.checked && !fp_ckpt_option_female.checked && !fp_ckpt_option_male.checked) {
        return window.errorModal("Please select the FastPitch checkpoint to fine-tune from", queueItemConfigModalContainer)
    }
    if (!hifigan_ckpt_option_male.checked && !hifigan_ckpt_option_female.checked ) {
        return window.errorModal("Please select the HiFi-GAN checkpoint to fine-tune from", queueItemConfigModalContainer)
    }
    if (fp_ckpt_option_other.checked && !trainingAddConfigCkptPathInput.value.trim().length) {
        return window.errorModal("You need to specify which FastPitch checkpoint to resume training from (fine-tuning only is supported, right now)", queueItemConfigModalContainer)
    }
    // if (!trainingAddConfigHiFiCkptPathInput.value.trim().length) {
    //     return window.errorModal("You need to specify which HiFi-GAN checkpoint to resume training from (fine-tuning only is supported, right now)", queueItemConfigModalContainer)
    // }
    if (!trainingAddConfigBatchSizeInput.value.trim().length) {
        return window.errorModal("Please enter the base batch size you'd like to use", queueItemConfigModalContainer)
    }
    if (!trainingAddConfigEpochsPerCkptInput.value.trim().length) {
        return window.errorModal("Please enter how many epochs to train for, between outputting checkpoints", queueItemConfigModalContainer)
    }

    const finishUp = () => {
        queueItemConfigModalContainer.style.display = "none"

        let fastpitch_checkpoint = "[male]"
        if (fp_ckpt_option_female.checked) {
            fastpitch_checkpoint = "[female]"
        } else if (fp_ckpt_option_other.checked) {
            fastpitch_checkpoint = trainingAddConfigCkptPathInput.replaceAll(/\\/, "/")
        }

        // TODO
        if (configAnExistingItem) {

            const queueIndex = window.training_state.currentlyConfiguringDatasetI

            const configData = {
                "dataset_path": window.training_state.datasetsQueue[queueIndex].dataset_path.replaceAll(/\\/, "/"),
                "output_path": trainingAddConfigOutputPathInput.value.replaceAll(/\\/, "/"),
                "checkpoint": fastpitch_checkpoint,
                "hifigan_checkpoint": hg_ckpt.replaceAll(/\\/, "/"),

                "use_amp": trainingAddConfigUseAmp.checked ? "true" : "false",
                "num_workers": parseInt(trainingAddConfigWorkersInput.value),
                "batch_size": parseInt(trainingAddConfigBatchSizeInput.value),
                "epochs_per_checkpoint": parseInt(trainingAddConfigEpochsPerCkptInput.value),
                "force_stage": trainingAddConfigDoForceStageCkbx.checked ? trainingAddConfigForceStageNumberSelect.value : undefined
            }

            configData.status = window.training_state.datasetsQueue[queueIndex].status

            window.training_state.datasetsQueue[queueIndex] = configData

        } else {

            const configData = {
                "status": "Ready",

                "dataset_path": trainingAddConfigDatasetPathInput.value.replaceAll(/\\/, "/"),
                "output_path": trainingAddConfigOutputPathInput.value.replaceAll(/\\/, "/"),
                "checkpoint": fastpitch_checkpoint,
                "hifigan_checkpoint": hg_ckpt.replaceAll(/\\/, "/"),

                "use_amp": trainingAddConfigUseAmp.checked ? "true" : "false",
                "num_workers": parseInt(trainingAddConfigWorkersInput.value),
                "batch_size": parseInt(trainingAddConfigBatchSizeInput.value),
                "epochs_per_checkpoint": parseInt(trainingAddConfigEpochsPerCkptInput.value),
                "force_stage": trainingAddConfigDoForceStageCkbx.checked ? trainingAddConfigForceStageNumberSelect.value : undefined
            }

            window.training_state.datasetsQueue.push(configData)
        }

        window.saveTrainingQueueJSON()
        window.refreshTrainingQueueList()
    }

    let fp_ckpt = trainingAddConfigCkptPathInput.value.trim().replaceAll(/\\/, "/")
    if (fp_ckpt_option_female.checked) {
        fp_ckpt = "[female]"
    } else if (fp_ckpt_option_male.checked) {
        fp_ckpt = "[male]"
    }
    // const hg_ckpt = trainingAddConfigHiFiCkptPathInput.value.trim().replaceAll(/\\/, "/")
    const hg_ckpt = hifigan_ckpt_option_male.checked ? "[male]" : "[female]"

    if (fp_ckpt!="[male]" && fp_ckpt!="[female]" && !fs.existsSync(fp_ckpt)) {
        window.confirmModal(`A FastPitch1.1 checkpoint file was not found at the following file/folder location. Continue regardless?<br>${fp_ckpt}`).then(resp => {
            if (resp) {
                setTimeout(() => {
                    if (hg_ckpt!="[male]" && hg_ckpt!="[female]" && !fs.existsSync(hg_ckpt)) {
                        window.confirmModal(`A HiFi-GAN checkpoint file was not found at the following file/folder location. Continue regardless?<br>${hg_ckpt}`).then(resp2 => {
                            if (resp2) {
                                finishUp()
                            }
                        })
                    } else {
                        finishUp()
                    }
                }, 500)
            }
        })
    } else {
        if (hg_ckpt!="[male]" && hg_ckpt!="[female]" && !fs.existsSync(hg_ckpt)) {
            window.confirmModal(`A HiFi-GAN checkpoint file was not found at the following file/folder location. Continue regardless?<br>${hg_ckpt}`).then(resp2 => {
                if (resp2) {
                    finishUp()
                }
            })
        } else {
            finishUp()
        }
    }
})



trainingQueueBtnAdd.addEventListener("click", () => {
    window.showConfigMenu()
    configAnExistingItem = false
})

trainingQueueBtnClear.addEventListener("click", () => {
    window.confirmModal("Are you sure you'd like to clear the training queue, losing all configured model training runs?").then(resp => {
        if (resp) {
            window.training_state.datasetsQueue = []
            window.refreshTrainingQueueList()
            fs.writeFileSync(`${window.path}/training_queue.json`, JSON.stringify(window.training_state.datasetsQueue, null, 4))
        }
    })
})

trainingOpenCkpts.addEventListener("click", () => {
    const output_path = window.training_state.datasetsQueue[window.training_state.selectedQueueItem].output_path
    if (fs.existsSync(output_path)) {
        const subItems = fs.readdirSync(output_path)
        if (subItems.length) {
            shell.showItemInFolder(`${output_path}/${subItems[0]}`)
        } else {
            shell.showItemInFolder(output_path)
        }
    }
})

trainingStartBtn.addEventListener("click", () => {
    if (window.userSettings.installation != "gpu") {
        window.errorModal("The CPU-only installation is enabled. Switch to the GPU installation if you can, or run training on a CUDA-compatible machine if not. It is not practical to run the training on the CPU, it would take forever.")
        return
    }

    window.ws.send(JSON.stringify({
        model: "",
        task: "startTraining",
        gpus: setting_cudaDevices.value.replaceAll(" ", ""),
        data: window.training_state.datasetsQueue[window.training_state.trainingQueueItem]
    }))
    trainingStartBtn.style.display = "none"
    trainingResumeBtn.style.display = "none"
    trainingPauseBtn.style.display = "block"
    trainingStopBtn.style.display = "block"
})
trainingPauseBtn.addEventListener("click", () => {
    trainingStartBtn.style.display = "none"
    trainingResumeBtn.style.display = "block"
    trainingPauseBtn.style.display = "none"
    window.ws.send(JSON.stringify({model: "", task: "pause"}))
})
trainingResumeBtn.addEventListener("click", () => {
    trainingStartBtn.style.display = "none"
    trainingPauseBtn.style.display = "block"
    trainingResumeBtn.style.display = "none"
    window.ws.send(JSON.stringify({model: "", task: "resume", data: window.training_state.datasetsQueue[window.training_state.trainingQueueItem]}))
})
trainingStopBtn.addEventListener("click", () => {
    trainingStartBtn.style.display = "block"
    trainingPauseBtn.style.display = "none"
    trainingResumeBtn.style.display = "none"
    trainingStopBtn.style.display = "none"
    window.training_state.isBatchTraining = false
    window.ws.send(JSON.stringify({model: "", task: "pause"}))
    setTimeout(() => {
        window.ws.send(JSON.stringify({model: "", task: "stop"}))
    }, 2000)
})

window._exportModelModalButton = createElem("button")
window.setupModal(window._exportModelModalButton, exportModelContainer)


btn_exportmodel.addEventListener("click", () => {
    if (addmissingmeta_btn_container.style.display == "flex") {
        return window.errorModal("Please first add the dataset metadata")
    }

    // Pre-fill with the checkpoints dir, if the dataset is already in the training queue
    const queueItem = window.training_state.datasetsQueue.filter(item => item.dataset_path.includes(window.appState.currentDataset))
    if (queueItem.length) {
        modelExport_trainningDir.value = queueItem[0].output_path+"/"+queueItem[0].dataset_path.split("/").reverse()[0]
    }
    const cachedOutPath = localStorage.getItem("modelExport_outputDir")
    if (cachedOutPath) {
        modelExport_outputDir.value = cachedOutPath
    }

    window._exportModelModalButton.click()
})
modelExport_outputDir.addEventListener("keyup", () => {
    localStorage.setItem("modelExport_outputDir", modelExport_outputDir.value.replaceAll(/\\/g, "/"))

})
exportSubmitButton.addEventListener("click", () => {

    if (!modelExport_trainningDir.value) {
        return window.errorModal("Please enter the output folder used during training, where all the checkpoints and intermediate files are")
    }
    if (!fs.existsSync(modelExport_trainningDir.value)) {
        return window.errorModal("The checkpoints directory was not found. Make sure you specify it from the drive letter. Eg C:/...")
    }
    if (!modelExport_outputDir.value) {
        return window.errorModal("Please enter where you want the model files exported to.")
    }
    if (!fs.existsSync(modelExport_outputDir.value)) {
        return window.errorModal("The export path was not found. Make sure you specify it from the drive letter. Eg C:/...")
    }
    let ckptFileFolder = `${modelExport_trainningDir.value.trim()}`
    if (!fs.existsSync(`${ckptFileFolder}/${window.appState.currentDataset}.pt`)) {
        if (fs.existsSync(`${modelExport_trainningDir.value.trim()}/${window.appState.currentDataset}/${window.appState.currentDataset}.pt`)) {
            ckptFileFolder = `${modelExport_trainningDir.value.trim()}/${window.appState.currentDataset}`
        } else {
            return window.errorModal("A FastPitch1.1 model file was not found in the given checkpoints directory. Have you trained it yet?")
        }
    }

    const doTheRest = () => {
        window.spinnerModal("Exporting...")

        // Copy over the resemblyzer embedding data, and export the .json metadata
        const trainingJSON = JSON.parse(fs.readFileSync(`${ckptFileFolder}/${window.appState.currentDataset}.json`, "utf8"))
        const metadataJSON = JSON.parse(fs.readFileSync(`${window.userSettings.datasetsPath}/${window.appState.currentDataset}/dataset_metadata.json`, "utf8"))
        const voiceId = metadataJSON.games[0].voiceId

        metadataJSON.games[0].resemblyzer = trainingJSON.games[0].resemblyzer
        metadataJSON.games[0].voiceId = voiceId//window.appState.currentDataset
        fs.writeFileSync(`${modelExport_outputDir.value.trim()}/${voiceId}.json`, JSON.stringify(metadataJSON, null, 4))

        // Copy over the model files
        fs.copyFileSync(`${ckptFileFolder}/${window.appState.currentDataset}.pt`, `${modelExport_outputDir.value.trim()}/${voiceId}.pt`)
        fs.copyFileSync(`${ckptFileFolder}/${window.appState.currentDataset}.hg.pt`, `${modelExport_outputDir.value.trim()}/${voiceId}.hg.pt`)

        doFetch(`http://localhost:${window.SERVER_PORT}/exportWav`, {
            method: "Post",
            body: JSON.stringify({
                fp_ckpt: `${ckptFileFolder}/${window.appState.currentDataset}.pt`,
                hg_ckpt: `${ckptFileFolder}/${window.appState.currentDataset}.hg.pt`,
                out_path: `${modelExport_outputDir.value.trim()}/${voiceId}.wav`
            })
        }).then(r=>r.text()).then((res) => {
            console.log("Exporting res:", res)

            if (res.length) {
                window.appLogger.log(res)
                window.errorModal(res)
                window.createModal("error", `There was an issue with exporting the preview audio file:<br><br>${res}`).then(resp => {
                    if (resp) {
                        shell.showItemInFolder(`${modelExport_outputDir.value.trim()}/${voiceId}.wav`)
                    }
                    exportModelContainer.click()
                })
            } else {
                window.confirmModal("Model exported successfully. Open output directory?").then(() => {
                    exportModelContainer.click()
                })
            }
        })
    }

    if (!fs.existsSync(`${ckptFileFolder}/${window.appState.currentDataset}.hg.pt`)) {
        return window.confirmModal("A HiFi-GAN model file was not found in the given checkpoints directory. Have you trained it yet?<br>(You can export anyway, without the audio preview or HiFi-GAN vocoder, but this is not yet ready for publishing, and the quality will be lower)").then(resp => {
            if (resp) {
                doTheRest()
            }
        })
    } else {
        doTheRest()
    }
})