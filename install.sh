
SUDO=
name=`which python3`
if [[ $name =~ local ]] ;
then
echo "Python3 is installed to local."
else
SUDO=sudo
fi

if [ "$(uname)" == 'Darwin' ]; then
  OS='MacOSX'
elif [ "$(expr substr $(uname -s) 1 5)" == 'Linux' ]; then
  OS='Linux'
  SUDO=sudo
else
  echo "Your platform ($(uname -a)) is not supported."
  exit 1
fi

echo "OS: $OS"
$SUDO pip3 uninstall -y acapture
$SUDO python3 setup.py install
python3 -c "import acapture"
