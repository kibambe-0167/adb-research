import cv2
import numpy as np
#first you create filters for each feature to extract, this is defined by the method below
def generate_face_feature_filters():
    filters = []

    # Edge detection filters
    orientations = [0, 45, 90, 135]
    scales = [0.5, 1.0, 1.5]
    for theta in orientations:
        theta *= np.pi / 180  # used to convert the angle from degrees to radians.
        for scale in scales:
            kernel = cv2.getGaborKernel((400, 400), scale, theta, 10.0, 0.5, 0.3, ktype=cv2.CV_32F)
            filters.append(kernel)

    # Texture filters
    orientations = [0, 45, 90, 135]
    scales = [1.0, 1.5, 2.0]
    for theta in orientations:
        theta *= np.pi / 4
        for scale in scales:
            kernel = cv2.getGaborKernel((400, 400), scale, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filters.append(kernel)

    # Eye detection filters
    orientations = [0, 45, 90, 135]
    scales = [0.5, 1.0, 1.5]
    for theta in orientations:
        theta *= np.pi / 120
        for scale in scales:
            kernel = cv2.getGaborKernel((200, 200), scale, theta, 10.0, 0.5, 1, ktype=cv2.CV_32F)
            filters.append(kernel)

    # # Nose detection filters
    orientations = [0, 45, 90, 135]
    scales = [0.5, 1.0, 1.5]
    for theta in orientations:
        theta *= np.pi / 180
        for scale in scales:
            kernel = cv2.getGaborKernel((31, 31), scale, theta, 10.0, 0.5, 0.2, ktype=cv2.CV_32F)
            filters.append(kernel)
    return filters

image_path = "Subset/000051.jpg" #add image path 

# Load the input image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Generate the face feature filters
face_filters = generate_face_feature_filters()

# Apply the face feature filters to the image
filtered_images = [] #storage mechanism
for kernel in face_filters:
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    filtered_images.append(filtered_image)

# Display the filtered images
for i, filtered_image in enumerate(filtered_images):
    cv2.imshow("Filter {}".format(i), filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
