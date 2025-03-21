import pyautogui

def handle_input(input_string):
    if input_string == 'angry':
        pyautogui.write('i')  # angry일 경우 "i" 입력
    elif input_string == 'sad':
        pyautogui.write('y')  # sad일 경우 "y" 입력
    else:
        print("Unknown input!")

# 예시 사용
user_input = input("Enter 'angry' or 'sad': ").strip().lower()
handle_input(user_input)
