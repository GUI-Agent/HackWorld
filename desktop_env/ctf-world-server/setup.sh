sudo apt install python3-pyatspi wmctrl
python3 -m venv --system-site-packages pyenv
source pyenv/bin/activate
pip install -r requirements.txt

sudo apt install gnome-screenshot
sudo cp ctfworld_server.service /etc/systemd/system/

sudo systemctl daemon-reexec
sudo systemctl daemon-reload

sudo systemctl enable ctfworld_server.service
sudo systemctl start ctfworld_server.service

sudo systemctl status ctfworld_server.service
