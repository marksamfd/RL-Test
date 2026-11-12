# bind the USB device on windows side and attach it to WSL
cmd.exe /C usbipd bind --busid 1-3
cmd.exe /C usbipd attach --wsl --busid 1-3

# Play camera video with ffplay
ffplay /dev/video0

# get what's using camera replace * by camera idx
fuser -v /dev/video*

# list devices to find camera idx
ls /dev