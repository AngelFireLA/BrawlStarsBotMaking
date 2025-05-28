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
