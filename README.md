## Fashion-app

# How to run this application

Clone the project
```bash
   git clone https://github.com/Chandradithya8/fashion-app.git
```
Build the docker image
```bash
   docker build -t your-image-name .
``
Run docker container
```bash
   docker run -p $port:80 you_image_name
``
Run the container on your localhost
```bash
   http://localhost:$port/
``
Go to /docs to try out the api
Click "Try it out" and upload an image and click "Execute" to get the response
