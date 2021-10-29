"use strict"


const smi = require('node-nvidia-smi')
smi((err, data) => {console.log(data.nvidia_smi_log.gpu.fb_memory_usage)})