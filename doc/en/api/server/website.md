# Start with website

This document provides the necessary steps to set up and run the web service for this project.

## 1. Starting the Web Service

### 1.1. Compiling the Web Code

Before you can compile the web code, make sure you have installed [Node.js](https://nodejs.org) version 18.3 or higher

Note: The version of Node.js in the Ubuntu or Debian GNU/Linux software repository is too low, causing compilation errors. Users can also install Node.js through the Nodesource repository, provided they uninstall the outdated version first.

```bash

  # sudo apt-get remove nodejs npm -y && sudo apt-get autoremove -y
  sudo apt-get update -y && sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/nodesource.gpg
  sudo chmod 644 /usr/share/keyrings/nodesource.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_23.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
  sudo apt-get update -y
  sudo apt-get install nodejs -y

```

Once npm is installed, navigate to the `ktransformers/website` directory:

```bash
cd ktransformers/website
```

Next, install the Vue CLI with the following command:

```bash
npm install @vue/cli
```

Now you can build the project:

```bash
npm run build
```
Finally you can build ktransformers with website:
```
cd ../../
pip install .
```
