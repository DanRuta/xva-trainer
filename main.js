
const PRODUCTION = process.mainModule.filename.includes("resources")
const path = PRODUCTION ? "resources/app" : "."

const fs = require("fs")
const {shell, app, BrowserWindow, ipcMain} = require("electron")
const {spawn} = require("child_process")

let pythonProcess

if (PRODUCTION) {
    // pythonProcess = spawn(`${path}/cpython/server.exe`, {stdio: "ignore"})
} else {
    pythonProcess = spawn("python", [`${path}/server.py`], {stdio: "ignore"})
}

let mainWindow
let discordClient
let discordClientStart = Date.now()

const createWindow = () => {

    mainWindow = new BrowserWindow({
        width: 1200,
        height: 1000,
        minHeight: 850,
        minWidth: 1200,
        frame: false,
        webPreferences: {
            nodeIntegration: true,
            // contextIsolation: false
        },
        icon: `${__dirname}/assets/x-icon.png`
    })

    mainWindow.loadFile("index.html")
    mainWindow.shell = shell

    mainWindow.on("ready-to-show", () => {
        mainWindow.show()
    })

    mainWindow.on("closed", () => mainWindow = null)
}

ipcMain.on("resize", (event, arg) => {
    mainWindow.setSize(arg.width, arg.height)
})
ipcMain.on("updatePosition", (event, arg) => {
    const bounds = mainWindow.getBounds()
    bounds.x = parseInt(arg.details[0])
    bounds.y = parseInt(arg.details[1])
    mainWindow.setBounds(bounds)
})
ipcMain.on("updateDiscord", (event, arg) => {

    if (!discordClient) {
        discordClient = require('discord-rich-presence')('890952337177321482') // xVATrainer
    }

    discordClient.updatePresence({
        state: 'Training AI voices',
        // details: arg.details,
        startTimestamp: discordClientStart,
        largeImageKey: 'xvatrainer_512_512',
        largeImageText: "xVATrainer",
        smallImageKey: 'xvatrainer_512_512',
        smallImageText: "xVATrainer",
        instance: true,
    })
})

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on("ready", createWindow)


// Quit when all windows are closed.
app.on("window-all-closed", () => {
    // On OS X it is common for applications and their menu bar
    // to stay active until the user quits explicitly with Cmd + Q
    if (process.platform !== "darwin") {
        app.quit()
    }
})

app.on("activate", () => {
// On OS X it"s common to re-create a window in the app when the
// dock icon is clicked and there are no other windows open.
    if (mainWindow === null) {
        createWindow()
    }
})