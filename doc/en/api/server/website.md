# Start with website

This document provides the necessary steps to set up and run the web service for this project.

## 1. Starting the Web Service

### 1.1. Compiling the Web Code

Before you can compile the web code, make sure you have installed [Node.js](https://nodejs.org) version 18.3 or higher

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
