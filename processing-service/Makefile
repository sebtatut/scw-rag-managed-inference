# Define variables
IMAGE_NAME = processing-service
CONTAINER_NAME = processing-service-container
PORT = 8000

# Default target
all: build run

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

# Stop the existing container if it is running
stop:
	@echo "Stopping existing container..."
	-docker stop $(CONTAINER_NAME) || true

# Remove the existing container if it exists
remove:
	@echo "Removing existing container..."
	-docker rm $(CONTAINER_NAME) || true

# Run the new Docker container
run: stop remove
	@echo "Running new Docker container..."
	docker run -d --env-file .env -p $(PORT):8000 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Full update target
update: build run

# Clean up all (container and image)
clean: stop remove
	@echo "Removing Docker image..."
	-docker rmi $(IMAGE_NAME) || true

.PHONY: all build stop remove run update clean
