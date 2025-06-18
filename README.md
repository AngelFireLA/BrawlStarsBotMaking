# BrawlStarsBotMaking
All the ressources you would need to create your very own external brawl stars bot.

In this repository, you will be finding basic guides and other resources that would help you code your own Brawl Stars Bot.

Notes :

- This is meant to help make an **external** Brawl Stars bot, which means this won’t help reverse engineering, accessing memory, injecting, or anything like that, in case you planned to do a dodgebot or an aimbot. (They are possible but much more difficult and it’s not the type of bot we’ll be focusing on). An external bots relies on getting data from an external point of view, like screenshots for example, and uses external means to interact with the game, like we will see later.  
- Knowing Python : The ressources here will primarily be intended for Python, and even though some of the things can be used with other languages, always assume we’re talking about Python.  
- Having a computer : An external Brawl Stars bot is maybe possible on mobile, but it’s very unlikely it would use python, and in general it’s more difficult because of the restrictions of Android phones (and I won’t even start talking about iOS devices).

This is, at the time of writing (28/05/2025), only my knowledge, so feel free to contact me on Discord (link at the end) or make a pull request.

# Setting up the basics

## Setting up the environment

### Python

First, if not already done, you need to [download and install Python](https://www.python.org/downloads/).  
Then, you will need an IDE, which is “an advanced text editor” that makes coding way easier.  
The most popular ones are :

- PyCharm (Python only)  
- Visual Studio Code Community (multi-languages)

I also recommend you learn about [virtual environments](https://www.w3schools.com/python/python_virtualenv.asp), which are isolated Python instances with their own libraries, very useful to speed up starting time and keeping the different projects separated.

Once you have your project on your IDE and your environment, you’re ready to start making your bot.

### Brawl Stars

To make a Brawl Stars bot, you will need somewhere to actually play Brawl Stars.  
The easiest and most common way is using an Android Emulator such as for example:

- [LDPlayer](https://ldplayer.net/)  
- [Bluestacks](https://www.bluestacks.com/fr/index.html)

The most important aspect is that the emulator supports being able to turn key presses on your keyboard into actions on the emulator, like pressing somewhere, moving a joystick, etc… As the other alternative will be much more difficult.

Once you have your emulator, just install Brawl Stars on it. You can use the Play Store, or any other app store, or even install Brawl Stars using its APK.

## Setting up Input/Output for your bot

### Input

First, you want to be able to get information about what’s happening in Brawl Stars so the bot can do whatever you want.  
For that, we will be using screenshots. 

Python has many screenshotting libraries, with all their own advantages and disadvantages (I will also give you snippets of code on how to do a full screen capture, if you want to capture a specific part of the screen you’ll have to research it based on the library you chose) :

- [PyAutoGUI](https://pypi.org/project/PyAutoGUI/) : This library is useful not only for taking screenshots, but also to click or handle the keyboard. It’s one of the easiest to use, but it’s by far one of the slowest, so it’s only recommended if you’re not very familiar with Python or if speed isn’t the priority for you yet.  
  ```python  
  import pyautogui  
  screenshot = pyautogui.screenshot()  
  ```  
    
- [Mss](https://pypi.org/project/mss/) : This library is a good compromise between speed, ease of use, and stability.  
  ```python  
  import mss  
  with mss.mss() as sct:  
      img = sct.grab(sct.monitors[1])   
  ```  
  This will take a full capture of your primary screen.  
- [BetterCam](https://pypi.org/project/bettercam/): It’s one of the fastest screenshotting libraries. It’s a bit longer to set up, and the main issue is that it’s not the most stable, sometimes the screen capture failing and returns Nothing, but even by taking into account the retries needed, it’s really fast.  
  For PylaAI for example, if we detect a capture failed, we just retry in a `while` loop until it works.   
  Or you could use another library as backup, for example “if screen capture fails, try again with Mss”.  
  To use it, you’ll first need to create a “camera” which is what you’ll be using to take screen captures whenever you need one.  
  ```python  
  import bettercam  
  camera = bettercam.create()  
    
  camera.grab() # this is to take a screen capture  
  ```  
- [PyWin32](https://pypi.org/project/pywin32/) : This library uses a more fundamental usage of the Windows API, which allows it to be really efficient when correctly used. You can use it to take screenshots, move the mouse, click, use the keyboard, and other things, but it’s more complicated to use, and at least for screenshots, isn’t the fastest.


To process your screenshots you will need image handling libraries. I recommend OpenCV as it’s faster than another popular alternative Pillow, and it’s installed by default if you install Bettercam) with it you can crop, resize, recolor, etc… your images easily.  
To install it `pip install opencv-pyton` and import it with `import cv2`.

### Output

Now that you have your images, it will be your job to extract information you want from them and then pick actions.  
To control your emulator, the easiest way is to use Key-Mappings, which is making specific keyboard keys do specific actions in game. For example with PylaAI :  
![image](https://github.com/user-attachments/assets/bd5ce983-a56f-4e36-8e7f-391bff936685)

Those key-mappings make it so whenever the E key is pressed, it presses on the super button (there is no key on the attack button because when the super isn’t ready, the super button automatically acts as the attack button).  
More than just a simple key->button, you can see on the left that most emulators also allow keys to be used for joysticks.

So, to make your bot actually do stuff, you just have to pick what key you want to do what, and make your code press that key when you want.  
To do that, you once again have a few options :

- PyAutoGUI : I already talked about it before, pyautogui is slow but can also be used to handle the keyboard :  
  ```python  
  import pyautogui  
    
  pyautogui.press('a') # Presses the 'a' key  
    
  pyautogui.write('hello world') # type a string  
    
  pyautogui.keyDown('shift')  
  pyautogui.press('a') # hold keys  
  pyautogui.keyUp('shift')  
  ```  
- [Keyboard](https://pypi.org/project/keyboard/) : It’s much faster and allows you to do the same things as PyAutoGUI :  
  ```python  
  import keyboard  
  keyboard.press_and_release('a')  
  keyboard.write('hello world')  
  keyboard.press('a')  
  keyboard.release('a')  
  ```


Some people also talked about using ADB to send inputs directly to the emulator, but I’m not knowledgeable enough to know if it’s possible without drawbacks, and if yes how, so unless someone expands on this section, we will stay with keyboard-only inputs.

# Using the Screenshots

## Knowing what part of the game you are in

You can now get screenshots of the emulator and you know you want to use that information to turn into keyboard/mouse presses, so the first thing is to actually know what you’re looking at, if you’re in the lobby, in a battle, waiting for a battle to start, etc…

### Template Matching

The easiest way is to use **Template Matching**. It’s simple, you take an image of your choice, and you check if it’s inside a given image.  
For example, to know if you’re in the Lobby, you could look for the brawler menu button icon inside the screenshot.  
Someone wanted to make a bot that would tell them if they’re still in matchmaking, and you could use Template Matching to easily do that, by searching for the “Exit” button (when it’s still red, as when it isn’t anymore, it means we found a match).

For that, you can use the library OpenCV which I mentioned before, as it has a function specially for that.  
Here’s a simple function that checks if an image (the template) is in the screenshot :  
```python  
confidence_min = 0.8 # confidence is between 0-1 and the closer to 1, the more sure the predictions need to be to count, because sometimes there could be something that doesn’t match exactly so it’s not sure  
def is_template_in_screenshot(template, screenshot):  
    result = cv2.matchTemplate(template, screenshot, cv2.TM_CCOEFF_NORMED)  
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  
    if max_val > confidence_min:  
        return True  
    else:  
        return False  
```

The template image would be an image you saved locally and loaded somewhere else in your script.

Tip : You would preferably crop the screenshot before so it only looks in the area you know the template is supposed to be, because it makes it faster and you’re less likely to find something else that would cause a false detection. 

You can use OpenCV to crop an image too :   
```python  
# you load opencv, your screenshot etc…  
# if we consider (x1, y1) the coordinates of the top left corner, and (x2, y2) the coordinates of the bottom right corner, of the area of the image you want to get  
x1, y1, x2, y2 = 0, 0, 100, 100  
cropped_image = screenshot[y1:y2, x1:x2] #returns the are between the corners (0, 0) and (100, 100)  
```

### OCR

Another way to know where you are is to use OCR.   
OCR is detecting text within an image.

It’s useful because you could use it to detect if the bot won or lost a match by detecting the texts “Victory” or “Defeat” or “Draw” or you can use it in the brawler menu to find the position of the icon of the brawler you’re looking for. (you might have to scroll, for that, using a keyboard, you can press the mouse, move it, and then release the mouse, and it will act as a scroll).

There are 2 main libraries that you could use (both libraries don’t work perfectly with the brawl stars text but it should be good enough and the first one you could try make it better yourself):

- Pytesseract : tesseract is the most well known OCR library, it is very configurable, you can quite easily train it to recognize a specific font, but it’s configurability is also a downside because it takes a lot of tries to find the best settings, and it’s really slow so I don’t recommend it if speed is your priority.  
- - Easyocr : Like it’s name says, easyocr is a libraries that allow OCR without a lot of setup, is quite fast (especially with gpu version of the torch library, we’ll talk about that later in the additional tips) and works well enough with Brawl Stars (it is what Pyla uses).  
  It’s best to recognize usual letters and numbers.  
  ```python  
  Import easyocr  
    
  reader = easyocr.Reader(['en']) # this is the object you’ll use to extract the text, just define it once at the top of your script  
  # you get your screenshot…  
  results = reader.readtext('screenshot.png')  
  # results is a list of tuples (bounding box, text, confidence) with bounding box being the four corners of the area of the text [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  
    
  # to get the text you could do simply   
  text_result = results[0][1] # if you expect the text only on one line  
  ```

## Detecting Enemies and other things in battle

### Object Detection

The most reliable external way to things entities such as player, allies, enemies, pets, etc… is using Object Detection. More precisely, it’s using AI vision models to find objects (player, enemy, etc..) inside a given image.

The problem is that if we find one of those models and give it a screenshot of Brawl Stars, it won’t find anything because it wasn’t trained to find Brawl Stars things.

A model’s **classes** is the list of of things a specific model will detect. A basic model could just have the “cat” class because it’s looking for cats only.

So we’ll have to either use a Brawl Stars specialized model or we’ll have to train our own model.  
To be clear, it would be much more difficult to make a global Brawl Stars model, it’s more stable to make a model dedicated to detect a specific type of thing.  
For example, there’s a model that can detect entities such as player/enemies/allies, but there’s also a separate model to detect the different wall tiles.

#### Using an existing model

Training your own model is a lot of work, so when you’re starting it’s recommended to use one of the models already made by the community, which are inside this github repository in the models folder.

Currently, all the models are YOLO models, usually YOLOv8 or YOLOv11 because they have a dedicated library, **ultralytics**, which makes them the easiest to use and train.

Models can be under two file extensions : .pt and .onnx :

- .pt are the original models that are used with ultralytics.  
- .onnx models are compiled versions of the .pt models that are a bit more difficult to use but they are slightly faster, and with the correct libraries they can be used by nearly all types of GPUs (that aren’t too old) without needing any modification.

I personally recommend using .pt files unless you know what you’re doing with the .onnx files.

Let’s now see how to actually use them inside a python script. We will first see .pt models, and then for .onnx models we will give you an example python class that’s able to use them similarly to .pt models, with some differences we’ll talk about later.

##### .pt models

First you will need the ultralytics python library.

Then, assuming you have a variable named “model_path” which is a string containing the model’s relative or absolute file path (look it up on google if you don’t know what I mean, you’ll need it for other things).  
Then it’s straightforward loading the model :  
```python  
from ultralytics import YOLO  
model_path = r“your_model_path”  
model = YOLO(model_path)  
```  
The r in front of the “ of model_path makes it so Python doesn’t annoy you because a path can have back-slashes,  in case you copy the path from the windows explorer.

To test the model, you’ll first need to get the image you want to detect things on.  
You could load them from a file with   
```python  
#...  
image_path = r'...'  
image = cv2.imread(image_path)  
```  
Or if you’re getting it from another source, be sure that the images’ colors aren’t inverted, because it’s something that cause me a lot of pain, using images that were BGR when they’re supposed to be RGB (means the red and blue pixel values were swapped).  
To check, you can just save the image to a file with  
```python  
# you get your image variable…  
cv2.imwrite(r“the path of where you want to save the image”, image)  
```  
Once you have your image, you can do for example :  
```python  
results = model.predict(source=image, conf=0.6, verbose=False, device='gpu')[0]  
```  
The only necessary parameter is source but the others are useful too :

- Verbose : by default for each time you use the model, it will say something in the console, but if you’re using it many times per second it would just be spam  
- device=’gpu’ : if you have the gpu version of the torch library (more on that later) and you have a gpu, you can make it use the gpu for much faster detections. Other device value would be ‘cpu’  
- Conf : For each detection, the model will assign a score between 0 and 1, and it’s how sure it is about it’s detection (like, how sure it is that the thing it detected is an enemy) so the conf parameter is “what’s the minimum confidence where I should keep a detection” it depends on the model, but usually a better trained model would have higher confidence in its detections.

results will contain a list of [Results](https://docs.ultralytics.com/modes/predict/#working-with-results) objects, but we only take the first element because when giving a single image there will always have only one element.  
You can easily get those detections as a list of dicts (where each element is one detection, for example one enemy or one tile) with  
```python  
results_data = json.loads(result.to_json())  
```  
Each of these elements/dicts have 2 very interesting keys :

- “name” : The name of the object detected (because you’ll probably need to different things depending on what object is detected)  
- “box” : A dict of the detected object’s coordinate as a rectangle  
  - “x1”  
  - “x2”  
  - “y1”  
  - “x2”

Here’s an example on how you would get a list of objects and their center coordinates :  
```python  
detections = []  
for item in results_data:  
    box = item["box"]  
    if not box:  
        continue  
    center_x = (box['x1'] + box['x2']) * opposite_scale_factor / 2  
    center_y = (box['y1'] + box['y2']) * opposite_scale_factor / 2  
    detections.append((item['name'], (center_x, center_y)))  
```  
You can then use the results whenever you want to in your logic.

#### Making your own model

To make your own Vision model, you will need 2 things in most cases : a dataset and a base model.

A dataset is a list of labeled images, which means a list of images where you also know exactly the coordinates of every element you want to detect.  
For example, if you want to make a model that detects cats, you need a lot of images containing cats but also where the cat is on that image. To tell “where the cat is” you would usually “draw a rectangle that englobes the cat” which is what we call a bounding box. You can draw a more detailed polygon in some specific cases (it depends on the model type, which we’ll see later), but as of writing this, we’ll consider only bounding boxes.  
The format of how you store the information depends on how you’ll train the model.

Let’s now see what images to put in your dataset :  
In your dataset, you will need images that contain what you want to detect, but also images that don't contain it, and images that contain what you want to detect, but also things that are similar but you don’t want to detect.  
Let me explain : if we want to make a model that detects cats, but we give it only images that contain cats, then it might learn that there’s at least one cat in every image, even if there aren’t.  
Another example for the part “also things that are similar”, would be that if on your images there are only cats and no other images, it might think that a dog is a cat because it’s “close enough” and it was never told that things like that wouldn’t be cats.

In short : Your dataset should contain images with what you want to detect, images without the thing you want to detect, images with/without what you want to detect and with things you want the model not to detect.

Once you have your images, you want to label them with the information about the objects inside.  
There are many apps and websites that you can use to label your images, such as Label Studio or Roboflow.

Roboflow is the site I would recommend as it has a lot of useful features, you can easily import your images, and label them inside the site, and then export the dataset in a lot of different formats depending on the model you want to train. They also provide you with a Google Collab Notebook to train your model so you barely need any coding knowledge.

Now, imagine you have your dataset, and you want to train a model, where do you start ?   
First, you need to find a base model. Because you realistically won’t create the model architecture yourself and everything that goes around that.  
The base model I recommend is a YOLO type model, which is the easiest to train and use, and is the most reliable I know of.  
For YOLO models, the labels are stored as .txt files with the same name as the image file.  
Your folder structure needs to resemble this :  
![pycharm64_64GTrwmTJi](https://github.com/user-attachments/assets/b202eb6d-da1d-4952-acab-56ba0b6e9d19)

Every image in the train or val folder needs to have their .txt counterpart in the train or val folder of the labels folder.  
Train folder is for the images that your model will actually see while training, while the Val folder contains image that the model will not see and instead they will be used to evaluate the model on images it hasn’t seen before. On 1000 images, you usually keep 100-300 images for evaluation.

Now let’s imagine those two folders are inside a dataset folder, the simplest way to train a YOLO model is using the ultralytics package (for YOLOv8 and YOLOv11) :  
```python  
   results = model.train(  
        data='dataset',  
        epochs=100,  
        imgsz=640,  
        batch=20,  
    )  
```  
Data being the path to the folder containing your dataset, epochs is “the amount of times your model will see every image” and usually the more the better, up to a certain point where it either doesn’t learn anything new, or worse, it does something called “overfitting”.  
Overfitting is when a model becomes really good on the data it was trained of but it’s not very good when it sees images it hasn’t seen before, which makes the model pretty much useless.  
You can usually see it because while training you will see the accuracy of the model on the training dataset and on the val dataset, and if the accuracy of the training dataset is much bigger, then it’s overfitting. (You can always ask chatgpt)

The imgsz parameter is the size of your images, because YOLO only supports square images (it will do the pre-processing of images automatically no matter the images you give it).  
Having a resolution too low will make it harder for the model to learn because it will be missing details, while making it too high can also make the training harder because there are more details so there can be more distracting details, and also it will make training much slower and require more vram/ram.  
640 is the default value and is usually good enough.  
Batch is how many images the model will train on at the same time. It helps it generalize more and also speeds up training, but how high you can put that depends on your ram and your GPU’s vram, so just try making it higher until you see it use too much of either. (Training without a GPU isn’t recommended).  
There are other parameters that you can research on your own, but in the end the model will be in runs/run_name/weights as last.pt

Congrats you now have trained your model. I recommend you test it on images it wasn’t trained on and compare the results with previous models to be sure it actually improved.

# Additional Info

## Tips

### Torch GPU version

If you have an Nvidia GPU, you can get better performance out of most models (especially ultralytics .pt ones) by installing a gpu-specific version of the torch library.  
You can get the installation command from their website depending on your settings :  
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)   
![chrome_Uxxlei0Yds](https://github.com/user-attachments/assets/c47b6adf-976a-4563-9a37-9175246cd03a)

**Note :** You will need to install CUDA and Cudnn on your computer for it to work on Windows, look it up online.

### PyGetWindow

PyGetWindow is a useful python library that allows you to get information about open windows and also control them.  
For example if you want to get the LDPlayer window :  
```python  
import pygetwindow  
window = pygetwindow.getWindowsWithTitle('LDPlayer')[0]  
```  
Using that window you can get useful information such as it’s x and y position, it’s width, it’s height, which are useful if you want to use relative coordinates in your bot when having to click or look for something.  
You can also use it to make a specific window, to minimize it, to maximize it, to be sure it’s in the good state :  
```python  
if window.isMinimized:  
  window.restore()

try:  
  window.activate()  
except: # if the window is already active, it would give an error  
  pass

window.maximize()  
```  
This is how PylaAI makes sure the app is visible and active and maximized.  
The one inconvenient thing is that you don’t have any access over if the window is in f11 full screen mode.

### Color Detection

Color detection is looking for a specific range of color on a screenshot and using that to detect information, usually with something called HSV masking.  
It’s mostly suggested when wanting to detect the player/allies/opponents by detecting the colored circles under the brawlers. That’s how PylaAI used to detect entities before switching to Object Detection, because even if it’s way faster, it’s less stable.

An example of usage in Pyla is detecting when the player gets idle disconnected because it’s only when there’s an error message that there’s a lot of a specific type of gray pixel.  
It’s also used to detect if the gadget or hypercharge is ready by looking for the specific green/purple pixels in a small area (the smaller the area you can reliably use, the more accurate it gets, because there’s less pixels that could be distracting the thing we want to find).  

## Libraries Starter Pack
See Python Libraries starter pack.txt for what I would recommend as libraries to make a good bot.

## Example
THe PylaAI source code is available if you want to look at how itn works :
https://github.com/PylaAI/PylaAI

## Contains
- A text file containing all the best libraries to use in my opinion (some speed has been sacrificed for some simplicity and for non-windows users)
- A python script to easily test your .pt models on images.

# Contributing 

If you want to contribute to this repository, there's a few things you can do :
- help in providing more information a topic mentionned here
- bring new information or tools or other things that weren't mentionned here
- do one of the things I'll mention in the TodoList
- submit a new mother or other ressource file

## Todo
- Convert the information on this page into a GitHub Wiki for better organisation
- Add examples and pratical use of color detection
- give examples of other types of models other than YOLO
- show how to use .onnx models
- start a section about Reinforcement Learning (some ideas : what it is, the basic terms, how it could work in brawl stars, some algorithms that could work, ideas, etc...)
- start a section about the different libraries to be used to make a menu (customtkinter, pygame, PyQT5 etc...)

# Current Available Models
| Model Name | Model Base | Model Type | Author | Classes | File Extensions | Date Uploaded |
|------------|------------|------------|--------|---------|------------------|----------------|
|PylaEntityDetectorV1|YOLOv8|entity detection|iyordanov| ['enemy', 'player', 'teammate'] | .pt and .onnx |16/06/2025|
|PylaEntityDetectorV2|YOLOv11|entity detection|REDACTED| ['enemy', 'teammate', 'player'] | .pt and .onnx |16/06/2025|
|PylaSpecificBrawlerDetectorV1|YOLOv8|specific brawler detection and wall detection |iyordanov| ['ammo', 'ball', 'damage_taken', 'defeat', 'draw', 'enemy_health_bar', 'enemy_position', 'gadget', 'gem', 'hypercharge', 'player_health_bar', 'player_position', 'respawning', 'shot_success', 'super', 'teammate_health bar', 'teammate_position', 'victory', 'wall', 'bush', '8bit', 'amber', 'ash', 'barley', 'bea', 'belle', 'bibi', 'bo', 'bonnie', 'brock', 'bull', 'buster', 'buzz', 'byron', 'carl', 'charlie', 'chester', 'chuck', 'colette', 'colt', 'cordelious', 'crow', 'darryl', 'doug', 'dynamike', 'edgar', 'primo', 'emz', 'eve', 'fang', 'frank', 'gale', 'gene', 'grom', 'gray', 'griff', 'gus', 'hank', 'jacky', 'janet', 'jessie', 'kit', 'larry_lawrie', 'leon', 'lola', 'lou', 'mandy', 'maisie', 'max', 'meg', 'melodie', 'mico', 'mortis', 'mrp', 'nani', 'nita', 'otis', 'pam', 'penny', 'piper', 'poco', 'rico', 'rosa', 'rt', 'ruffs', 'sam', 'sandy', 'shelly', 'spike', 'sprout', 'stu', 'squeak', 'surge', 'tara', 'tick', 'willow', 'lily', 'enemy_ability', 'ally_ability'] | .pt and .onnx |16/06/2025|
|PylaWallDetectorV1|YOLOv11|wall detection|angelfire| ["bush", "blue_post", "block_of_stone&grass", "cubic_wall", "wooden_barrel", "wooden_box", "wooden_box", "wooden_fence", "invincible_wall", "cactus", "potted_plant" ,"yellow_wall", "blue_barrel", "metallic_fence", "blue_weights"] | .pt and .onnx |16/06/2025|

To submit a model, make a pull request containing your model in the correct folder (or in a new model type folder if it doesn't fit in any current one) and also add a line to the Available Models Table with the necessary info

# Join PylaAI discord to talk to other Devs and maybe join a potential Brawl Stars Bot competition : https://discord.com/invite/ehMRX9hpFN 
