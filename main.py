import json
import cv2
import face_recognition
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import time
import sys
from threading import Thread
import queue
import numpy as np
import getpass
import msvcrt
import os
from datetime import datetime
from pynput import keyboard
from PIL import Image
import pyautogui
import webbrowser
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from tkinter import filedialog
import tkinter as tk


# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create usage directory if it doesn't exist
if not os.path.exists('usage'):
    os.makedirs('usage')

def get_formatted_date():
    """Get formatted date for log file name"""
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
             'July', 'August', 'September', 'October', 'November', 'December']
    now = datetime.now()
    month = months[now.month - 1]
    day = str(now.day)
    if day.endswith('1') and day != '11':
        day += 'st'
    elif day.endswith('2') and day != '12':
        day += 'nd'
    elif day.endswith('3') and day != '13':
        day += 'rd'
    else:
        day += 'th'
    return f"{month}_{day}_{now.year}"

def log_conversation(role, message):
    """Log conversation to file with timestamp"""
    timestamp = datetime.now().strftime('%I:%M:%S %p')  # 12-hour format with AM/PM
    log_file = f"logs/Conversation_{get_formatted_date()}.txt"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        # Add divider before user input
        if role == "User":
            f.write("\n" + "-" * 50 + "\n")
        
        # Write the message
        f.write(f"[{timestamp}] {role}: {message}\n")
        
        # Add divider after Gemo's response
        if role == "Gemo":
            f.write("-" * 50 + "\n")

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize Gemini API
genai.configure(api_key=config['gemini_api_key'])
# Initialize two models - one for chat and one for vision
chat_model = genai.GenerativeModel("gemini-1.5-flash")
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize conversation history
conversation_history = []

# Initialize chat
chat = chat_model.start_chat(history=[])

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if config['voice_settings']['voice'] in voice.name:
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    """Function to convert text to speech"""
    global engine
    try:
        engine = pyttsx3.init()
        for voice in engine.getProperty('voices'):
            if config['voice_settings']['voice'] in voice.name:
                engine.setProperty('voice', voice.id)
                break
        engine.say(text)
        engine.runAndWait()
    except:
        # If text-to-speech fails, just print
        print(f"(Voice): {text}")



def clear_terminal(show_help=False):
    """Clear the terminal screen and optionally show help message"""
    os.system('cls' if os.name == 'nt' else 'clear')
    if show_help:
        help_message = """Hello! I'm Gemo. How can I help you today?
Tip: Use 'analyze: path/to/image.jpg' to analyze images!
Use 'usage' to see API usage statistics.
System Commands:
• 'open <app>' - Open applications (browser, notepad, spotify, etc.)
• 'search <query>' - Search Google
• 'play <query>' - Play videos on YouTube
• 'cls' or 'clear' or 'clean' - Clear terminal
• '/help' - Show this help message
• 'stop' - Stop current speech"""
        print(help_message)

class MicrophoneManager:
    def __init__(self):
        self.is_listening = False
        self.should_stop = False
        self.recognizer = sr.Recognizer()
        self.mic = None
        try:
            self.mic = sr.Microphone()
            
        except Exception as e:
            print("\nVoice input is not available. Using text input only.")
    
    def start_listening(self, speech_queue):
        """Start listening for speech input"""
        if not self.mic:
            return
        
        self.should_stop = False
        self.is_listening = True
        
        while not self.should_stop:
            try:
                with self.mic as source:
                    try:
                        audio = self.recognizer.listen(source, timeout=0.1)
                        text = self.recognizer.recognize_google(audio)
                        speech_queue.put(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError:
                        continue
            except Exception as e:
                time.sleep(0.1)
                if self.should_stop:
                    break
        
        self.is_listening = False
    
    def stop_listening(self):
        """Stop listening for speech input"""
        self.should_stop = True
        while self.is_listening:
            time.sleep(0.1)

def password_verification():
    """Fallback password verification"""
    print("\nCamera not available. Using password verification.")
    speak("Camera not available. Using password verification.")
    
    attempts = 3
    while attempts > 0:
        password = getpass.getpass("Enter password: ")
        if password == config['fallback_password']:
            speak("Password correct. Access granted.")
            print("Access granted!")
            return True
        attempts -= 1
        if attempts > 0:
            print(f"Incorrect password. {attempts} attempts remaining.")
            speak(f"Incorrect password. {attempts} attempts remaining.")
    
    speak("Access denied. Too many failed attempts.")
    print("Access denied. Too many failed attempts.")
    return False

def face_verification():
    """Function to perform face verification"""
    speak("Initializing biometric scan")
    print("Initializing biometric scan...")
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend for Windows
        if not cap.isOpened():
            print("Could not access camera")
            cap.release()
            return password_verification()
        
        # Load authorized face encodings
        authorized_encodings = []
        for face_path in config['authorized_faces']:
            try:
                # Read image with OpenCV first
                auth_image = cv2.imread(face_path)
                if auth_image is None:
                    speak("Error loading authorized face image")
                    print(f"Error loading image: {face_path}")
                    return password_verification()
                
                # Ensure image is 8-bit RGB
                auth_image = cv2.cvtColor(auth_image, cv2.COLOR_BGR2RGB)
                auth_image = np.array(auth_image, dtype=np.uint8)
                
                # Get face encodings
                face_locs = face_recognition.face_locations(auth_image)
                if not face_locs:
                    speak("No face found in the authorized image")
                    print(f"No face found in: {face_path}")
                    return password_verification()
                    
                auth_encoding = face_recognition.face_encodings(auth_image, face_locs)[0]
                authorized_encodings.append(auth_encoding)
            except Exception as e:
                print(f"Error processing authorized face image: {str(e)}")
                return password_verification()
        
        for attempt in range(2):  # Two attempts for verification
            if attempt > 0:
                speak("Initializing secondary biometric scan")
                print("Initializing secondary biometric scan...")
            
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error capturing frame from camera")
                continue
            
            try:
                # Convert camera frame to 8-bit RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.array(frame, dtype=np.uint8)
                
                # Find faces in the frame
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(frame, face_locations)
                    
                    # Check if any face matches
                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(authorized_encodings, encoding)
                        if True in matches:
                            cap.release()
                            cv2.destroyAllWindows()
                            speak("Access granted")
                            print("Access granted!")
                            return True
                else:
                    print("No face detected in camera frame")
            
            except Exception as e:
                print(f"Error processing camera frame: {str(e)}")
            
            if attempt == 0:
                time.sleep(2)  # Wait before second attempt
        
        speak("Access denied. Unauthorized user detected")
        print("Access denied. Unauthorized user detected.")
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    except Exception as e:
        print("Could not access camera")
        return password_verification()

def get_input_with_mic_control(mic_manager, speech_queue, speech_thread, timeout=None):
    """Get input while controlling microphone state based on typing"""
    buffer = ""
    prompt = "User: "
    print(f"\n{prompt}", end='', flush=True)
    
    # Initialize start_time if timeout is provided
    start_time = time.time() if timeout else None
    
    while True:
        # Start listening if microphone is available and buffer is empty
        if mic_manager.mic and not speech_thread and not buffer:
            speech_thread = Thread(target=mic_manager.start_listening, args=(speech_queue,), daemon=True)
            speech_thread.start()
        
        # Check for speech input if no text is being typed
        if not buffer and speech_thread:
            try:
                speech_input = speech_queue.get_nowait()
                print(f"(Speech Input: {speech_input})")
                return speech_input, None
            except queue.Empty:
                pass
        
        # Check for keyboard input
        if msvcrt.kbhit():
            char = msvcrt.getwche()
            
            # Handle Enter key
            if char == '\r':
                print()  # Move to next line
                return buffer, None
            
            # Handle backspace - only if buffer has content
            elif char == '\b':
                if buffer:
                    buffer = buffer[:-1]
                    print(' \b', end='', flush=True)  # Clear the character
                else:
                    # If buffer is empty, reprint the prompt to prevent deletion
                    print(prompt[-1], end='', flush=True)
            
            # Handle regular characters
            else:
                buffer += char
                
                # Stop microphone as soon as we start typing
                if len(buffer) == 1 and speech_thread:
                    mic_manager.stop_listening()
                    speech_thread = None

        # Check for timeout
        if timeout and time.time() - start_time > timeout:
            return buffer, None

def analyze_image(image_path, prompt=None):
    """Analyze image using Gemini Vision API"""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return f"Error: Image file '{image_path}' not found."
        
        # Open and prepare image
        image = Image.open(image_path)
        
        # Prepare the prompt
        if not prompt:
            prompt = "Describe this image in detail. What do you see?"
        
        # Generate response
        response = vision_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

class APIUsageTracker:
    def __init__(self):
        self.usage_file = f"usage/api_usage_{get_formatted_date()}.txt"
        self.chat_requests = 0
        self.vision_requests = 0
        
        # Free tier Gemini 1.5 Flash limits
        self.requests_per_minute = 15  # RPM (Requests Per Minute)
        self.tokens_per_minute = 1_000_000  # TPM (Tokens Per Minute)
        self.requests_per_day = 1_500  # RPD (Requests Per Day)
        
        # Token limits
        self.input_token_limit = 1_048_576  # ~1M tokens
        self.output_token_limit = 8_192  # ~8K tokens
        
        self.load_usage()
    
    def load_usage(self):
        """Load existing usage data if available"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        self.chat_requests = int(lines[0].split(': ')[1])
                        self.vision_requests = int(lines[1].split(': ')[1])
        except Exception:
            pass
    
    def save_usage(self):
        """Save current usage data"""
        with open(self.usage_file, 'w') as f:
            f.write(f"Chat Requests: {self.chat_requests}\n")
            f.write(f"Vision Requests: {self.vision_requests}\n")
    
    def increment_chat(self):
        """Increment chat API usage"""
        self.chat_requests += 1
        self.save_usage()
    
    def increment_vision(self):
        """Increment vision API usage"""
        self.vision_requests += 1
        self.save_usage()
    
    def get_usage_percentages(self):
        """Calculate usage percentages"""
        total_requests = self.chat_requests + self.vision_requests
        usage_percent = (total_requests / self.requests_per_day) * 100
        return usage_percent
    
    def get_remaining_requests(self):
        """Calculate remaining requests"""
        total_requests = self.chat_requests + self.vision_requests
        remaining = self.requests_per_day - total_requests
        return remaining
    
    def display_usage(self):
        """Return usage statistics string"""
        total_requests = self.chat_requests + self.vision_requests
        usage_percent = self.get_usage_percentages()
        remaining = self.get_remaining_requests()
        
        return f"\nAPI Usage Today ({get_formatted_date()}):\n" + \
               f"Total Requests: {total_requests:,} / {self.requests_per_day:,} " + \
               f"({usage_percent:.1f}% used)\n" + \
               f"Remaining Requests: {remaining:,}\n\n" + \
               f"Rate Limits (Free Tier):\n" + \
               f"• {self.requests_per_minute} requests per minute\n" + \
               f"• {self.tokens_per_minute:,} tokens per minute\n" + \
               f"• {self.requests_per_day:,} requests per day\n\n" + \
               f"Token Limits:\n" + \
               f"• Input: {self.input_token_limit:,} tokens\n" + \
               f"• Output: {self.output_token_limit:,} tokens"

# Initialize API usage tracker
usage_tracker = APIUsageTracker()

class EmailManager:
    def __init__(self):
        # Load credentials
        with open('credentials.json', 'r') as f:
            self.credentials = json.load(f)
        
        # Load contacts
        with open('contacts.json', 'r') as f:
            self.contacts = json.load(f)
    
    def get_email_address(self, name):
        """Get email address from contact name"""
        name = name.lower()
        return self.contacts.get(name)
    
    def send_email(self, to_email, body, subject=None, attachments=None):
        """Send email with optional subject and attachments"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.credentials['email']
            msg['To'] = to_email
            msg['Subject'] = subject if subject else "No Subject"
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments if any
            if attachments:
                for file_path in attachments:
                    with open(file_path, 'rb') as f:
                        part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
                        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                        msg.attach(part)
            
            # Connect to server and send
            with smtplib.SMTP(self.credentials['smtp_server'], self.credentials['smtp_port']) as server:
                server.starttls()
                server.login(self.credentials['email'], self.credentials['password'])
                server.send_message(msg)
            
            return True, "Email sent successfully!"
        except Exception as e:
            return False, f"Error sending email: {str(e)}"

# Initialize email manager
email_manager = EmailManager()

class SystemAutomation:
    def __init__(self):
        self.known_apps = {
            "spotify": lambda: (webbrowser.open("spotify:"), "Opening Spotify"),
            "discord": lambda: (subprocess.Popen(r"C:\Users\welcome\AppData\Local\Discord\Discord.exe"), "Opening Discord"),
            "notepad": lambda: (subprocess.Popen("notepad.exe"), "Opening Notepad"),
            "chrome": lambda: (pyautogui.hotkey('win', 'r') or time.sleep(0.2) or pyautogui.write('chrome') or pyautogui.press('enter'), "Opening Chrome"),
            "browser": lambda: (pyautogui.hotkey('win', 'r') or time.sleep(0.2) or pyautogui.write('chrome') or pyautogui.press('enter'), "Opening Browser"),
            "edge": lambda: (subprocess.Popen(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"), "Opening Microsoft Edge"),
            "recycle bin": lambda: (subprocess.run(['explorer', 'shell:RecycleBinFolder']), "Opening Recycle Bin"),
            "bin": lambda: (subprocess.run(['explorer', 'shell:RecycleBinFolder']), "Opening Recycle Bin"),
            "calculator": lambda: (subprocess.Popen("calc.exe"), "Opening Calculator"),
            "calc": lambda: (subprocess.Popen("calc.exe"), "Opening Calculator"),
            "microsoft edge": lambda: (subprocess.Popen(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"), "Opening Microsoft Edge"),
            "microsoft store": lambda: (subprocess.run(['explorer', r'shell:Appsfolder\Microsoft.WindowsStore_8wekyb3d8bbwe!App']), "Opening Microsoft Store")
        }
    
    def is_valid_url(self, url):
        """Check if text looks like a URL"""
        return '.' in url and ' ' not in url
    
    def open_link(self, url):
        """Open a URL in the default browser"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            webbrowser.open(url)
            return True
        except:
            return False
    
    def open_app(self, command):
        """Process open command for apps and URLs"""
        command = command.replace("open ", "").strip()
        items_to_open = command.split()  # Splitting the command into words
        opened_any = False
        current_phrase = ""
        response = None

        for item in items_to_open:
            if current_phrase:
                current_phrase += " " + item
            else:
                current_phrase = item
            
            item_lower = current_phrase.lower().strip()
            
            if item_lower in self.known_apps:
                action, message = self.known_apps[item_lower]()
                response = message
                opened_any = True
                current_phrase = ""  # Reset phrase after opening an app
            elif self.is_valid_url(current_phrase):
                success = self.open_link(current_phrase)
                if success:
                    response = f"Opening {current_phrase}"
                    opened_any = True
                current_phrase = ""  # Reset phrase after opening a URL
            else:
                if len(items_to_open) == 1 or self.is_valid_url(item):
                    response = f"Unrecognized application or invalid URL: {item}"
                    current_phrase = ""  # Reset for any unrecognized item

        # Only notify if nothing valid was opened
        if not opened_any:
            response = "Sorry, I couldn't process any of the specified items."
        
        return response
    
    def search_google(self, query):
        """Search on Google"""
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            return f"Searching Google for: {query}"
        except Exception as e:
            return f"Error searching Google: {str(e)}"
    
    def play_youtube(self, query):
        """Search and play on YouTube or open playlist"""
        try:
            # Check if it's just "music"
            if query.lower().strip() == "music":
                webbrowser.open(config['youtube_playlist'])
                return "Opening your music playlist"
            
            # Otherwise search on YouTube
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            time.sleep(2)  # Wait for page to load
            pyautogui.press('tab', presses=10, interval=0.1)  # Navigate to first video
            pyautogui.press('enter')
            return f"Playing on YouTube: {query}"
        except Exception as e:
            return f"Error playing YouTube video: {str(e)}"
    
    def handle_command(self, command):
        """Handle various system commands"""
        command = command.lower().strip()
        
        # Check for specific commands
        if command.startswith('open '):
            return self.open_app(command)
        elif command.startswith('search '):
            query = command[7:].strip()
            return self.search_google(query)
        elif command.startswith('play '):
            query = command[5:].strip()
            return self.play_youtube(query)
        elif command.startswith('email '):
            return self.handle_email(command[6:].strip())
        
        return None  # Not a system command
    
    def handle_email(self, recipient_name):
        """Handle email sending process"""
        # Get recipient email
        recipient_email = email_manager.get_email_address(recipient_name)
        if not recipient_email:
            return f"Sorry, I couldn't find {recipient_name} in contacts."
        
        # Get email body
        print("\nGemo: What would you like to email?")
        speak("What would you like to email?")
        
        # Initialize speech recognition for body
        speech_queue = queue.Queue()
        mic_manager = MicrophoneManager()
        speech_thread = None
        body, _ = get_input_with_mic_control(mic_manager, speech_queue, speech_thread)
        
        # Get subject
        print("\nGemo: Boss Would you like to add a subject?")
        speak("Boss Would you like to add a subject?")
        
        # Initialize speech recognition for subject
        speech_queue = queue.Queue()
        mic_manager = MicrophoneManager()
        speech_thread = None
        subject_input, _ = get_input_with_mic_control(mic_manager, speech_queue, speech_thread)
        
        subject = None
        if subject_input and subject_input.lower().startswith('subject will be:'):
            subject = subject_input[15:].strip()
        
        # Check for attachments
        print("\nGemo: Would you like to add attachments?")
        speak("Would you like to add attachments? Sir")
        
        # Initialize speech recognition for attachments
        speech_queue = queue.Queue()
        mic_manager = MicrophoneManager()
        speech_thread = None
        attachment_response, _ = get_input_with_mic_control(mic_manager, speech_queue, speech_thread)
        
        attachments = None
        if attachment_response and any(word in attachment_response.lower() for word in ['yes', 'sure', 'attach', 'files will be']):
            # Create and hide root window
            root = tk.Tk()
            root.withdraw()
            
            # Open file dialog
            attachments = filedialog.askopenfilenames(
                title="Select files to attach",
                filetypes=[("All files", "*.*")]
            )
            
            if attachments:
                print(f"Selected files: {', '.join(os.path.basename(f) for f in attachments)}")
        
        # Send email
        success, message = email_manager.send_email(recipient_email, body, subject, attachments)
        return message

# Initialize system automation
system_automation = SystemAutomation()

class SpeechManager:
    def __init__(self):
        self.engine = None
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.speech_thread = None
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize the text-to-speech engine"""
        try:
            if self.engine:
                self.engine.endLoop()
        except:
            pass
        
        self.engine = pyttsx3.init()
        with open('config.json', 'r') as f:
            config = json.load(f)
        for voice in self.engine.getProperty('voices'):
            if config['voice_settings']['voice'] in voice.name:
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak_in_thread(self):
        """Handle speech queue in a separate thread"""
        while True:
            try:
                if not self.speech_queue.empty():
                    text = self.speech_queue.get()
                    self.is_speaking = True
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.is_speaking = False
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.initialize_engine()
                time.sleep(0.1)
    
    def speak(self, text):
        """Add text to speech queue"""
        if not self.speech_thread or not self.speech_thread.is_alive():
            self.speech_thread = Thread(target=self.speak_in_thread, daemon=True)
            self.speech_thread.start()
        
        self.speech_queue.put(text)
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.is_speaking:
            try:
                self.engine.stop()
            except:
                pass
            self.initialize_engine()
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
            self.is_speaking = False

# Initialize speech manager globally
speech_manager = SpeechManager()

def speak(text):
    """Global function to speak text"""
    speech_manager.speak(text)

def stop_speaking():
    """Global function to stop speaking"""
    speech_manager.stop_speaking()

def handle_stop_command(command):
    """Check if command is to stop speaking"""
    stop_commands = ['stop', 'shutup', 'quiet', 'silence', 'shut up', 'be quiet']
    return any(command.lower().strip() == cmd for cmd in stop_commands)

def main():
    # Perform face verification
    if not face_verification():
        sys.exit(1)
    
    # Initialize speech recognition
    speech_queue = queue.Queue()
    mic_manager = MicrophoneManager()
    speech_thread = None
    
    # Clear terminal and show initial help message
    clear_terminal(show_help=True)
    
    while True:
        user_input, speech_thread = get_input_with_mic_control(mic_manager, speech_queue, speech_thread)
        
        if user_input:  # Only process if we have input (text or speech)
            try:
                # Check for stop command first
                if handle_stop_command(user_input):
                    stop_speaking()
                    continue
                
                # Log user input
                log_conversation("User", user_input)
                
                # Check for clear commands
                user_input_lower = user_input.lower().strip()
                if user_input_lower in ['cls', 'clear', 'clean']:
                    clear_terminal(show_help=False)
                    continue
                elif user_input_lower == '/help':
                    clear_terminal(show_help=True)
                    continue
                
                # Stop current speech before processing new command
                stop_speaking()
                
                # Check for system commands
                system_response = system_automation.handle_command(user_input)
                if system_response:
                    response_text = system_response
                # Then check for other commands
                elif user_input.lower().strip() == 'usage':
                    response_text = usage_tracker.display_usage()
                elif user_input.lower().startswith(("analyze:", "see:")):
                    parts = user_input.split(":", 1)[1].strip().split("|")
                    image_path = parts[0].strip()
                    prompt = parts[1].strip() if len(parts) > 1 else None
                    response_text = analyze_image(image_path, prompt)
                    usage_tracker.increment_vision()
                else:
                    # Regular chat
                    conversation_history.append({"role": "user", "parts": [user_input]})
                    response = chat.send_message(user_input)
                    response_text = response.text
                    usage_tracker.increment_chat()
                
                # Add AI response to history and log it
                conversation_history.append({"role": "assistant", "parts": [response_text]})
                log_conversation("Gemo", response_text)
                
                print(f"\nGemo: {response_text}")
                speak(response_text)
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                print(f"\nGemo: {error_msg}")
                speak(error_msg)
                log_conversation("Error", error_msg)

if __name__ == "__main__":
    main() 