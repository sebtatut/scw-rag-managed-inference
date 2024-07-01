# Use the official Nginx image from the Docker Hub
FROM nginx:latest

# Copy the custom Nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Copy static files to the Nginx HTML directory
COPY static /usr/share/nginx/html

# Expose port 80
EXPOSE 80
