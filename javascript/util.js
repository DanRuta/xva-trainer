/**
 * String.prototype.replaceAll() polyfill
 * https://gomakethings.com/how-to-replace-a-section-of-a-string-with-another-one-with-vanilla-js/
 * @author Chris Ferdinandi
 * @license MIT
 */
if (!String.prototype.replaceAll) {
    String.prototype.replaceAll = function(str, newStr){

        // If a regex pattern
        if (Object.prototype.toString.call(str).toLowerCase() === '[object regexp]') {
            return this.replace(str, newStr);
        }

        // If a string
        return this.replace(new RegExp(str, 'g'), newStr);

    };
}

window.toggleSpinnerButtons = () => {
    const spinnerVisible = window.getComputedStyle(spinner).display == "block"
    spinner.style.display = spinnerVisible ? "none" : "block"
    keepSampleButton.style.display = spinnerVisible ? "block" : "none"
    generateVoiceButton.style.display = spinnerVisible ? "block" : "none"
    samplePlay.style.display = spinnerVisible ? "flex" : "none"
}

window.confirmModal = message => new Promise(resolve => resolve(createModal("confirm", message)))
window.spinnerModal = message => new Promise(resolve => resolve(createModal("spinner", message)))
window.errorModal = (message, keepModals=[]) => new Promise(resolve => resolve(createModal("error", message, keepModals)))
window.createModal = (type, message, keepModals=[]) => {
    return new Promise(resolve => {
        modalContainer.innerHTML = ""
        const displayMessage = message.prompt ? message.prompt : message
        const modal = createElem("div.modal#activeModal", {style: {opacity: 0}}, createElem("span", displayMessage))
        modal.dataset.type = type

        if (type=="confirm") {
            const yesButton = createElem("button")
            yesButton.innerHTML = "Yes"
            const noButton = createElem("button")
            noButton.innerHTML = "No"
            modal.appendChild(createElem("div", yesButton, noButton))

            yesButton.addEventListener("click", () => {
                closeModal(modalContainer, keepModals).then(() => {
                    resolve(true)
                })
            })
            noButton.addEventListener("click", () => {
                closeModal(modalContainer, keepModals).then(() => {
                    resolve(false)
                })
            })
        } else if (type=="error" || type=="error_nocopy") {
            const closeButton = createElem("button")
            closeButton.innerHTML = "Close"
            const buttonsContainer = createElem("div")
            buttonsContainer.appendChild(closeButton)

            if (type=="error") {
                const clipboardButton = createElem("button")
                clipboardButton.innerHTML = "Copy to clipboard"
                clipboardButton.addEventListener("click", () => {
                    clipboardButton.innerHTML = "Copied"
                    navigator.clipboard.writeText(displayMessage.replaceAll("<br>", "\n"))
                })
                buttonsContainer.appendChild(clipboardButton)
            }

            modal.appendChild(buttonsContainer)

            closeButton.addEventListener("click", () => {
                closeModal(modalContainer, keepModals).then(() => {
                    resolve(false)
                })
            })
        } else if (type=="prompt") {
            const closeButton = createElem("button")
            closeButton.innerHTML = "Submit"
            const inputElem = createElem("input", {type: "text", value: message.value})
            modal.appendChild(createElem("div", inputElem))
            modal.appendChild(createElem("div", closeButton))

            closeButton.addEventListener("click", () => {
                closeModal(modalContainer, keepModals).then(() => {
                    resolve(inputElem.value)
                })
            })
        } else {
            modal.appendChild(createElem("div.spinner", {style: {borderLeftColor: "#ccc"}}))
        }

        modalContainer.appendChild(modal)
        modalContainer.style.opacity = 0
        modalContainer.style.display = "flex"

        requestAnimationFrame(() => requestAnimationFrame(() => modalContainer.style.opacity = 1))
        requestAnimationFrame(() => requestAnimationFrame(() => chromeBar.style.opacity = 1))
    })
}
window.closeModal = (container=undefined, notThisOne=undefined) => {
    return new Promise(resolve => {
        const allContainers = [settingsContainer, container, modalContainer, toolsContainer, datasetMetaContainer, preprocessAudioContainer, preprocessTextContainer, cleanAudioTextContainer, trainContainer, queueItemConfigModalContainer]
        const containers = container==undefined ? allContainers : (Array.isArray(container) ? container.filter(c=>c!=undefined) : [container])

        notThisOne = Array.isArray(notThisOne) ? notThisOne : (notThisOne==undefined ? [] : [notThisOne])

        containers.forEach(cont => {
            // Fade out the containers except the exceptions
            if (cont!=undefined && !notThisOne.includes(cont)) {
                cont.style.opacity = 0
            }
        })

        const someOpenContainer = allContainers.filter(c=>c!=undefined).find(cont => cont.style.opacity==1 && cont.style.display!="none" && cont!=modalContainer)
        if (!someOpenContainer || someOpenContainer==container) {
            chromeBar.style.opacity = 0.88
        }

        setTimeout(() => {
            containers.forEach(cont => {
                // Hide the containers except the exceptions
                if (cont!=undefined && !notThisOne.includes(cont)) {
                    cont.style.display = "none"
                    const someOpenContainer2 = allContainers.filter(c=>c!=undefined).find(cont => cont.style.opacity==1 && cont.style.display!="none" && cont!=modalContainer)
                    if (!someOpenContainer2 || someOpenContainer2==container) {
                        chromeBar.style.opacity = 0.88
                    }
                }
            })
        }, 200)
        try {
            activeModal.remove()
        } catch (e) {}
        resolve()
    })
}


window.addEventListener("resize", e => {
    window.userSettings.customWindowSize = `${window.innerHeight},${window.innerWidth}`
    saveUserSettings()
})

// Keyboard actions
// ================
window.addEventListener("keyup", event => {
    if (!event.ctrlKey) {
        window.ctrlKeyIsPressed = false
    }
    if (!event.shiftKey) {
        window.shiftKeyIsPressed = false
    }
})

window.addEventListener("keydown", event => {

    if (event.ctrlKey && event.key.toLowerCase()=="r") {
        location.reload()
    }
    if (event.ctrlKey && event.shiftKey && event.key.toLowerCase()=="i") {
        window.electron = require("electron")
        electron.remote.BrowserWindow.getFocusedWindow().webContents.openDevTools()
        return
    }

    const key = event.key.toLowerCase()

    if (event.ctrlKey) {
        window.ctrlKeyIsPressed = true
    }
    if (event.shiftKey) {
        window.shiftKeyIsPressed = true
    }

    // The Enter key to submit text input prompts in modals
    if (event.key=="Enter" && modalContainer.style.display!="none" && event.target.tagName=="INPUT") {
        activeModal.querySelector("button").click()
    }

    if (window.ctrlKeyIsPressed && key=="s") {
        btn_save.click()
        return
    }
    if (!window.shiftKeyIsPressed && window.ctrlKeyIsPressed && key=="arrowdown") {
        btn_newline.click()
        return
    }
    if (window.ctrlKeyIsPressed && key=="f") {
        btn_play.click()
        return
    }
    if (window.ctrlKeyIsPressed && key=="x") {
        btn_delete.click()
        return
    }
    if (window.ctrlKeyIsPressed && key=="d") {
        btn_record.click()
        return
    }


    // Disable keyboard controls while in a text input
    if (event.target.tagName=="INPUT" || event.target.tagName=="TEXTAREA") {
        return
    }

    if (key==" ") {
        event.preventDefault()
        event.stopPropagation()
        return
    }

    if (key=="arrowup") {
        if (window.appState.currentDataset==undefined) return
        if (window.appState.recordFocus!=undefined) {
            window.setRecordFocus(Math.max(0, window.appState.recordFocus-1))
        } else {
            window.setRecordFocus(0)
        }
        return
    }
    if (key=="arrowdown") {
        if (window.appState.currentDataset==undefined) return

        const numRecords = window.datasets[window.appState.currentDataset].metadata.length
        if (window.appState.recordFocus!=undefined) {
            window.setRecordFocus(Math.min(numRecords-1, window.appState.recordFocus+1))
        } else {
            window.setRecordFocus(numRecords-1)
        }
        return
    }

    // if (event.target==textInput || event.target==letterPitchNumb || event.target==letterLengthNumb) {
    //     // Enter: Generate sample
    //     if (event.key=="Enter") {
    //         generateVoiceButton.click()
    //         event.preventDefault()
    //     }
    //     return
    // }


    // Escape: close modals
    if (key=="escape") {
        closeModal()
    }




    // Space: bring focus to the input textarea
    // if (key==" ") {
    //     setTimeout(() => textInput.focus(), 0)
    // }
    // CTRL-S: Keep sample
    // if (key=="s" && event.ctrlKey && !event.shiftKey) {
    //     keepSampleFunction(false)
    // }
    // // CTRL-SHIFT-S: Keep sample (but with rename prompt)
    // if (key=="s" && event.ctrlKey && event.shiftKey) {
    //     keepSampleFunction(true)
    // }
    // Y/N for prompt modals
    if (key=="y" || key=="n" || key==" ") {
        if (document.querySelector("#activeModal")) {
            const buttons = Array.from(document.querySelector("#activeModal").querySelectorAll("button"))
            const yesBtn = buttons.find(btn => btn.innerHTML.toLowerCase()=="yes")
            const noBtn = buttons.find(btn => btn.innerHTML.toLowerCase()=="no")
            if (key=="y") yesBtn.click()
            if (key==" ") yesBtn.click()
            if (key=="n") noBtn.click()
        }
    }
})



window.setupModal = (openingButton, modalContainerElem, callback) => {
    openingButton.addEventListener("click", () => {
        if (callback) {
            callback()
        }
        closeModal(undefined, modalContainerElem).then(() => {
            modalContainerElem.style.opacity = 0
            modalContainerElem.style.display = "flex"
            requestAnimationFrame(() => requestAnimationFrame(() => modalContainerElem.style.opacity = 1))
            requestAnimationFrame(() => requestAnimationFrame(() => chromeBar.style.opacity = 1))
        })
    })
    modalContainerElem.addEventListener("click", event => {
        if (event.target==modalContainerElem) {
            window.closeModal(modalContainerElem)
        }
    })
}


function CSVToArray( strData, strDelimiter ){
    // Check to see if the delimiter is defined. If not,
    // then default to comma.
    strDelimiter = (strDelimiter || ",");

    // Create a regular expression to parse the CSV values.
    var objPattern = new RegExp(
        (
            // Delimiters.
            "(\\" + strDelimiter + "|\\r?\\n|\\r|^)" +

            // Quoted fields.
            "(?:\"([^\"]*(?:\"\"[^\"]*)*)\"|" +

            // Standard fields.
            "([^\"\\" + strDelimiter + "\\r\\n]*))"
        ),
        "gi"
        );

    // Create an array to hold our data. Give the array
    // a default empty first row.
    var arrData = [[]];

    // Create an array to hold our individual pattern
    // matching groups.
    var arrMatches = null;

    // Keep looping over the regular expression matches
    // until we can no longer find a match.
    while (arrMatches = objPattern.exec( strData )){

        // Get the delimiter that was found.
        var strMatchedDelimiter = arrMatches[ 1 ];

        // Check to see if the given delimiter has a length
        // (is not the start of string) and if it matches
        // field delimiter. If id does not, then we know
        // that this delimiter is a row delimiter.
        if (
            strMatchedDelimiter.length &&
            strMatchedDelimiter !== strDelimiter
            ){
            // Since we have reached a new row of data,
            // add an empty row to our data array.
            arrData.push( [] );
        }

        var strMatchedValue;

        // Now that we have our delimiter out of the way,
        // let's check to see which kind of value we
        // captured (quoted or unquoted).
        if (arrMatches[ 2 ]){
            // We found a quoted value. When we capture
            // this value, unescape any double quotes.
            strMatchedValue = arrMatches[ 2 ].replace(
                new RegExp( "\"\"", "g" ),
                "\""
                );
        } else {
            // We found a non-quoted value.
            strMatchedValue = arrMatches[ 3 ];

        }

        // Now that we have our value string, let's add
        // it to the data array.
        arrData[ arrData.length - 1 ].push( strMatchedValue );
    }

    // Return the parsed data.
    return( arrData );
}
window.CSVToArray = CSVToArray


window.getNextFileName = dataset => {
    if (fs.existsSync(`${window.userSettings.datasetsPath}/${dataset}/wavs`)) {
        const numFiles = fs.readdirSync(`${window.userSettings.datasetsPath}/${dataset}/wavs`).length

        let nameNumber = numFiles
        let nameValid = false
        let maxIters = 50000
        let iters = 0

        while (!nameValid && iters<maxIters) {

            let name = `${nameNumber}`.padStart(6, "0")
            const existingRecords = window.datasets[dataset].metadata.filter(records => {
                return records[0].fileName.split(".")[0]==name
            })

            if (existingRecords.length==0) {
                nameValid = true
            } else {
                nameValid = false
                nameNumber += 1
            }
            iters += 1
        }

        if (iters>=maxIters) {
            console.warn("Max while iters reached")
        }

        return `${nameNumber}`.padStart(6, "0")

    } else {
        fs.mkdirSync(`${window.userSettings.datasetsPath}/${dataset}/wavs`)
        return "0".padStart(6, "0")
    }

}



window.formatSecondsToTimeString = totalSeconds => {
    const timeDisplay = []

    if (totalSeconds > (60*60)) { // hours
        const hours = parseInt(totalSeconds/(60*60))
        timeDisplay.push(hours+"h")
        totalSeconds -= hours*(60*60)
    }
    if (totalSeconds > (60)) { // minutes
        const minutes = parseInt(totalSeconds/(60))
        timeDisplay.push(String(minutes).padStart(2, "0")+"m")
        totalSeconds -= minutes*(60)
    }
    if (totalSeconds > (1)) { // seconds
        const seconds = parseInt(totalSeconds/(1))
        timeDisplay.push(String(seconds).padStart(2, "0")+"s")
        totalSeconds -= seconds*(1)
    }

    return timeDisplay.join(" ")
}

window.initFilePickerButton = (button, input, setting, properties, filters=undefined, defaultPath=undefined, callback=undefined) => {
    button.addEventListener("click", () => {
        const defaultPath = input.value.replace(/\//g, "\\")
        let filePath = electron.remote.dialog.showOpenDialog({ properties, filters, defaultPath})
        if (filePath) {
            filePath = filePath[0].replace(/\\/g, "/")
            input.value = filePath.replace(/\\/g, "/")
            if (setting) {
                setting = typeof(setting)=="function" ? setting() : setting
                window.userSettings[setting] = filePath
                saveUserSettings()
            }
            if (callback) {
                callback()
            }
        }
    })
}



// Disable page navigation on badly dropped file
window.document.addEventListener("dragover", event => event.preventDefault(), false)
window.document.addEventListener("drop", event => event.preventDefault(), false)

// https://stackoverflow.com/questions/18052762/remove-directory-which-is-not-empty
const path = require('path')
window.deleteFolderRecursive = function (directoryPath, keepRoot=false) {
if (fs.existsSync(directoryPath)) {
    fs.readdirSync(directoryPath).forEach((file, index) => {
      const curPath = path.join(directoryPath, file);
      if (fs.lstatSync(curPath).isDirectory()) {
       // recurse
        window.deleteFolderRecursive(curPath);
      } else {
        // delete file
        fs.unlinkSync(curPath);
      }
    });
    if (!keepRoot) {
        fs.rmdirSync(directoryPath);
    }
  }
};