"use strict"

const smi = require('node-nvidia-smi')

// https://www.chartjs.org/docs/latest/
window.updateTrainingGraphValues = (allValues, chart) => {
    chart.data.datasets[0].data = allValues.map(pairs => pairs[1])
    chart.data.labels = allValues.map(pairs => pairs[0])
    chart.update()
}

const startingData = []
window.loss_chart_object = window.initChart("loss_plot", "Loss", "255,112,67", {startingData: startingData, max: 100000, minY: null, boundsMax: null})
window.loss_delta_chart_object = window.initChart("loss_delta_plot", "Loss Delta", "255,112,67", {startingData: startingData, max: 100000, boundsMax: null, extradataset: {
        label: "Target",
        borderColor: `rgb(50,50,250)`,
        data: Array.from(Array(60)).map((_,i)=>0),
    }})


window.training_state = {
    // state: 0, // 0 idle, 1 training, 2 paused, 3 finished
    isBatchTraining: false,
    stage_viewed: 1,
    dirs_watching: [],
    datasetsQueue: [
        {
            voice_name: "MaleEvenToned",
            dataset_path: "H:/xVA/FP_INPUT/sk_maleeventoned",
            output_path: "H:/xVA/FP_OUTPUT/sk_maleeventoned",
            batch_size: 32,
            num_workers: 24,
            epochs_per_checkpoint: 5,
            force_stage: undefined,
            status: "Ready"
        }
    ],
    trainingQueueItem: 0,
    websocketDynamicLogLine: "",
    currentlyViewedDataset: "H:/xVA/FP_INPUT/sk_maleeventoned",
    selectedQueueItem: undefined,

}

// Load training batch queue file
try {
    fs.readFile(`${window.path}/training_queue.json`, "utf8", (err, data) => {
        if (data) {
            window.training_state.datasetsQueue = JSON.parse(data)
            window.refreshTrainingQueueList()
        }
    })
} catch (e) {}

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
                trainingStartBtn.disabled = false
                training_title.innerHTML = `Training - ${nameBox.title}`

                if (!fs.existsSync(dataset.output_path)) {
                    fs.mkdirSync(dataset.output_path)
                }

                startTrackingFolder(dataset.dataset_path, dataset.output_path)
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
            window.showConfigMenu(window.training_state.datasetsQueue[di], di)
        })
    })
}


window.training_state.trainingQueueItem = 0
window.training_state.stage_viewed = 1

window.trainingTextLogNumLines = 0
window.updateTrainingLogText = () => {
    try {
        const datasetConfig = window.training_state.datasetsQueue.find(ds => ds.dataset_path==window.training_state.currentlyViewedDataset)

        if (fs.existsSync(`${datasetConfig.output_path}/training.log`)) {
            const logData = fs.readFileSync(`${datasetConfig.output_path}/training.log`, "utf8").split("\n")
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
    if (fs.existsSync(`${datasetConfig.output_path}/graphs.json`)) {
        const jsonString = fs.readFileSync(`${datasetConfig.output_path}/graphs.json`, "utf8")

        try {
            const graphsJson = JSON.parse(jsonString)
            const stageData = graphsJson.stages[window.training_state.stage_viewed]
            window.updateTrainingGraphValues(stageData.loss, window.loss_chart_object)
            window.updateTrainingGraphValues(stageData.loss_delta, window.loss_delta_chart_object)



            // window.loss_delta_chart_object.data.datasets[1].data = stageData.loss_delta.map((_,i)=>0.04)
            if (stageData.target_delta) {
                window.loss_delta_chart_object.data.datasets[1].data = stageData.loss_delta.map((_,i)=>stageData.target_delta)
                window.loss_delta_chart_object.data.labels = stageData.loss_delta.map(pairs => pairs[0])
                window.loss_delta_chart_object.update()
            }


        } catch (e) {
            // console.log(e)
            // console.log(`${datasetConfig.output_path}/graphs.json`)
        }
    }
}


const startTrackingFolder = (dataset_path, output_path) => {
    if (!Object.keys(window.training_state.dirs_watching).includes(output_path)) {
        window.training_state.dirs_watching[output_path] = fs.watch(output_path, {recursive: false, persistent: true}, (eventType, fileName) => {

            if (window.training_state.currentlyViewedDataset==dataset_path && fileName=="training.log") {
                window.updateTrainingLogText()
            }

            if (window.training_state.currentlyViewedDataset==dataset_path && fileName=="graphs.json") {
                window.updateTrainingGraphs()
            }
        })
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




window.showConfigMenu = (startingData, di) => {

    queueItemConfigModalContainer.style.opacity = 1
    queueItemConfigModalContainer.style.display = "flex"

    const configData = (typeof startingData == "string" ? undefined : startingData) || {
        "status": "Ready",

        "dataset_path": undefined,
        "output_path": undefined,
        "checkpoint": "H:/xVA/FP_OUTPUT/sk_femaleeventoned/FastPitch_checkpoint_1760_85765.pt",

        "num_workers": 4, // TODO, app-level default settings
        "batch_size": 8, // TODO, app-level default settings
        "epochs_per_checkpoint": 3, // TODO, app-level default settings
        "force_stage": undefined,
    }
    if (typeof startingData == "string") {
        configData["dataset_path"] = startingData
    }

    window.training_state.currentlyConfiguringDatasetI = di

    trainingAddConfigDatasetPathInput.value = ""
    trainingAddConfigDatasetPathInput.disabled = !!startingData

    trainingAddConfigDoForceStageCkbx.checked = configData.force_stage!=undefined
    if (configData.force_stage!=undefined) {
        trainingAddConfigForceStageNumberSelect.value = configData.force_stage
        trainingAddConfigForceStageNumberSelect.disabled = false
    } else {
        trainingAddConfigForceStageNumberSelect.value = "1"
        trainingAddConfigForceStageNumberSelect.disabled = true
    }

    trainingAddConfigDatasetPathInput.value = configData.dataset_path || ""
    trainingAddConfigOutputPathInput.value = configData.output_path || ""
    trainingAddConfigCkptPathInput.value = configData.checkpoint

    trainingAddConfigWorkersInput.value = parseInt(configData.num_workers)
    trainingAddConfigBatchSizeInput.value = parseInt(configData.batch_size)
    trainingAddConfigEpochsPerCkptInput.value = parseInt(configData.epochs_per_checkpoint)

    queueItemConfigModalContainer.style.display = "flex"
}

trainingAddConfigDoForceStageCkbx.addEventListener("change", () => {
    trainingAddConfigForceStageNumberSelect.disabled = !trainingAddConfigDoForceStageCkbx.checked
})


let configAnExistingItem = false
cancelConfig.addEventListener("click", () => {
    queueItemConfigModalContainer.style.display = "none"
})
acceptConfig.addEventListener("click", () => {

    if (!trainingAddConfigDatasetPathInput.value.trim().length) {
        return window.errorModal("You need to specify where your dataset is located.", queueItemConfigModalContainer)
    }
    if (!trainingAddConfigOutputPathInput.value.trim().length) {
        return window.errorModal("You need to specify where to output the intermediate data/models for your dataset.", queueItemConfigModalContainer)
    }
    if (!trainingAddConfigCkptPathInput.value.trim().length) {
        return window.errorModal("You need to specify which checkpoint to resume training from (fine-tuning only is supported, right now)", queueItemConfigModalContainer)
    }
    if (!trainingAddConfigBatchSizeInput.value.trim().length) {
        return window.errorModal("Please enter the base batch size you'd like to use", queueItemConfigModalContainer)
    }
    if (!trainingAddConfigEpochsPerCkptInput.value.trim().length) {
        return window.errorModal("Please enter how many epochs to train for, between outputting checkpoints", queueItemConfigModalContainer)
    }

    queueItemConfigModalContainer.style.display = "none"

    // TODO
    if (configAnExistingItem) {

        const queueIndex = window.training_state.currentlyConfiguringDatasetI

        const configData = {
            "dataset_path": window.training_state.datasetsQueue[queueIndex].dataset_path,
            "output_path": trainingAddConfigOutputPathInput.value,
            "checkpoint": trainingAddConfigCkptPathInput.value,

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

            "dataset_path": trainingAddConfigDatasetPathInput.value,
            "output_path": trainingAddConfigOutputPathInput.value,
            "checkpoint": trainingAddConfigCkptPathInput.value,

            "num_workers": parseInt(trainingAddConfigWorkersInput.value),
            "batch_size": parseInt(trainingAddConfigBatchSizeInput.value),
            "epochs_per_checkpoint": parseInt(trainingAddConfigEpochsPerCkptInput.value),
            "force_stage": trainingAddConfigDoForceStageCkbx.checked ? trainingAddConfigForceStageNumberSelect.value : undefined
        }

        window.training_state.datasetsQueue.push(configData)
    }

    window.saveTrainingQueueJSON()
    window.refreshTrainingQueueList()
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

