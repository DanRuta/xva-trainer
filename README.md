# xVA Scribe

xVAScribe is a speech dataset creation tool. It serves two use cases: (1) recording a custom dataset via the microphone (2) Adding or editing transcripts for existing audio data. Automatic transcription is built into the app for use as a noisy first-pass for this.


xVAScribe was built as a companion tool for xVASynth, a machine learning speech synthesis app, using voices from characters in video games.




## Instructions


To start, download the latest release, and double click the `xVAScribe.exe` file. Alternatively, check out the `Development` section, to see how to run the non-packaged dev code.

You can initialize a new dataset in three ways: (1) completely fresh by clicking the + button at the top right. (2) from existing transcript data only, by drag+dropping a `.csv` or `.txt` file into the drop zone  (3) from existing audio and/or text data, by creating a folder in the `datasets` folder, placing a `metadata.csv` file (if you have a transcript), and `wavs` folder in it with the audio files, following the usual speech dataset formatting. You can check what this format looks like by first creating a couple of dummy lines in a fresh dataset. The `.txt` file structure needs to be \\n newline separated transcript lines.

A dataset can be loaded by clicking its name in the panel on the left. Lines can be edited by first selecting them in the main window

### Recording a new dataset via the microphone

You can record new audio for lines in your dataset, be it new lines you'll enter through the app, or existing lines in an imported dataset transcript (recommended).

You can select which microphone you wish to use in the settings menu (accessible via the settings cog at the top right). Also here, you can explore using automatic background noise removal and its strength, after first recording some noisy silence. Experiment with it on/off, and the strength to see what works best for your setup.

The app uses incremental numerical zero padded file names for new lines.

I recommend finding a transcript from an existing speech dataset, such as MAILABS or LJSpeech, and using one of their transcripts, to avoid having to devise a transcript content of your own. But feel free to add your own lines, if there is some unique vocabulary you'd like the model to be good at.

### Adjusting/writing new transcripts for existing audio data

All buttons are mapped to keyboard shortcuts to enable a good pace of getting through the work. To further help speed things up, the app has a built-in model for automatic speech recognition, to optionally get a noisy first pass on all audio. Clean this up where it doesn't work well. The model needs the audio to be mono and 22050Hz.




## Development

`npm install` dependencies, and run with `npm start`. To make changes to the ASR model script, use virtualenv, and `pip install -r requirements.txt` using Python 3.6+. Use pyinstaller and the spec file to package it up into an `.exe`.


## Packaging

First, run the scripts in `package.json` to create the electron distributables.


## Support and Future Plans

Check the Discord server for support, and chat about suggestions for future plans.
