# BLIP+CLIP: Open Clip Guided BLIP Captions

Generates a batch of BLIP captions and scores them using Open CLIP, then returns the sorted captions. Results in more consistently correct captions.

## Install
Clone the https://github.com/salesforce/BLIP repo, then copy this in. Make sure the other one works, run 
````pip install open_clip easyocr```` there might be more packages I've forgotten about, sorry.

## Run
````
python3 server.py
````

Send POST requests to http://localhost:5015/prompt with multipart formdata as the body type. Send the image as a file with the key 'image', read the source for the other arguments.

## Pony Fix
If you're using this for pony stuff, use pony_fix:true as one of your form settings, as for some reason, BLIP consistently labels ponies as "pinkies" and that applies a basic replace to the BLIP outputs. Despite how dumb it sounds, it works pretty well.
