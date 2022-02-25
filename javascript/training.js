"use strict"


const smi = require('node-nvidia-smi')
smi((err, data) => {console.log(data.nvidia_smi_log.gpu.fb_memory_usage)})


// For Disk usage, I can do something like:
// typeperf "\\DESKTOP-CV2U1LN\LogicalDisk(H:)\% Disk Time"
// https://stackoverflow.com/questions/52822588/how-to-get-calculate-the-current-disk-activity-in-node-js-on-windows


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

window.cpu_chart_object = initChart("cpu_chart", "CPU", "62,149,205")
window.cuda_chart_object = initChart("cuda_chart", "CUDA", "62,149,205")
window.ram_chart_object = undefined
window.vram_chart_object = undefined

window.disk_chart_object = initChart("disk_chart", "Disk", "140,200,130")



window.updateSystemGraphs = () => {

    const totalRam = window.totalmem()
    const ramUsage = window.totalmem() - window.freemem()
    const percentRAM = ramUsage/totalRam*100

    if (!window.ram_chart_object) {
        window.ram_chart_object = initChart("ram_chart", "RAM", "191,101,191", {boundsMax: totalRam/1024})
    }



    window.cpuUsage(cpuLoad => {
        smi((err, data) => {
            let totalVRAM
            let usedVRAM
            let computeLoad

            if (data.nvidia_smi_log.gpu.length) {
                totalVRAM = parseInt(data.nvidia_smi_log.gpu[0].fb_memory_usage.total.split(" ")[0])
                usedVRAM = parseInt(data.nvidia_smi_log.gpu[0].fb_memory_usage.used.split(" ")[0])
                computeLoad = parseFloat(data.nvidia_smi_log.gpu[0].utilization.gpu_util.split(" ")[0])
            } else {
                totalVRAM = parseInt(data.nvidia_smi_log.gpu.fb_memory_usage.total.split(" ")[0])
                usedVRAM = parseInt(data.nvidia_smi_log.gpu.fb_memory_usage.used.split(" ")[0])
                computeLoad = parseFloat(data.nvidia_smi_log.gpu.utilization.gpu_util.split(" ")[0])
            }
            const percent = usedVRAM/totalVRAM*100

            if (!window.vram_chart_object) {
                window.vram_chart_object = initChart("vram_chart", "VRAM", "62,149,205", {boundsMax: parseInt(totalVRAM/1024)})
            }


            let vramUsage = `${(usedVRAM/1000).toFixed(1)}/${(totalVRAM/1000).toFixed(1)} GB (${percent.toFixed(2)}%)`
            // console.log(`RAM: ${parseInt(ramUsage)}GB/${parseInt(totalRam)}GB (${percentRAM.toFixed(2)}%) | CPU: ${(cpuLoad*100).toFixed(2)}% | GPU: ${computeLoad}% | VRAM: ${vramUsage}`)


            // CPU
            cpu_chart_object.data.datasets[0].data.push(parseInt(cpuLoad*100))
            if (cpu_chart_object.data.datasets[0].data.length>60) {
                cpu_chart_object.data.datasets[0].data.splice(0,1)
            }
            cpu_chart_object.data.datasets[0].label = `${cpu_chart_object.data.datasets[0].label.split(" ")[0]} ${(cpuLoad*100).toFixed(2)}%`

            // CUDA
            cuda_chart_object.data.datasets[0].data.push(parseInt(computeLoad))
            if (cuda_chart_object.data.datasets[0].data.length>60) {
                cuda_chart_object.data.datasets[0].data.splice(0,1)
            }
            cuda_chart_object.data.datasets[0].label = `${cuda_chart_object.data.datasets[0].label.split(" ")[0]} ${computeLoad}%`

            // RAM
            ram_chart_object.data.datasets[0].data.push(parseInt(ramUsage/1024))
            if (ram_chart_object.data.datasets[0].data.length>60) {
                ram_chart_object.data.datasets[0].data.splice(0,1)
            }
            ram_chart_object.data.datasets[0].label = `${ram_chart_object.data.datasets[0].label.split(" ")[0]} ${(ramUsage/1024).toFixed(1)}GB/${(totalRam/1024).toFixed(1)}GB`

            // VRAM
            vram_chart_object.data.datasets[0].data.push(parseInt(usedVRAM/1024))
            if (vram_chart_object.data.datasets[0].data.length>60) {
                vram_chart_object.data.datasets[0].data.splice(0,1)
            }
            vram_chart_object.data.datasets[0].label = `${vram_chart_object.data.datasets[0].label.split(" ")[0]} ${(usedVRAM/1024).toFixed(1)}/${(totalVRAM/1024).toFixed(1)} GB (${percent.toFixed(2)}%)`

            cpu_chart_object.update()
            ram_chart_object.update()
            vram_chart_object.update()
            cuda_chart_object.update()
            window.updateSystemGraphs()


        })
    })
}
window.updateSystemGraphs()




trainingStartBtn.addEventListener("click", () => {
    window.ws.send(JSON.stringify({
        model: "",
        task: "startTraining",
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
    window.ws.send(JSON.stringify({model: "", task: "resume"}))
})
trainingStopBtn.addEventListener("click", () => {
    trainingStartBtn.style.display = "block"
    trainingPauseBtn.style.display = "none"
    trainingResumeBtn.style.display = "none"
    trainingStopBtn.style.display = "none"
    window.ws.send(JSON.stringify({model: "", task: "pause"}))
    setTimeout(() => {
        window.ws.send(JSON.stringify({model: "", task: "stop"}))
    }, 2000)
})