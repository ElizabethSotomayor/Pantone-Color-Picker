#import these libraries
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

#import these files
from colors import pantone_colors
from webcam import detect_color_from_webcam

# Variables to store clicked color
clicked_color = None
clicked_name = None
clicked_points = []

#Used to speed up processing time by randomly selecting pixels from the image
def sample_pixels(pixels, sample_size=50000):
   
    filtered_pixels = pixels

    # Sample from the filtered pixels
    if len(filtered_pixels) > sample_size:
        indices = np.random.choice(len(filtered_pixels), sample_size, replace=False)
        return filtered_pixels[indices]
    return filtered_pixels

#Used to segment the image via minibatch kmeans, faster than kmeans
def segment_image_and_sample_colors(image, n_clusters):
    """
    Segment the image using MiniBatchKMeans and extract sampled colors.
    """
    # Convert to Lab color space for better color separation
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    #flatten image for clustering
    pixels = lab_image.reshape((-1, 3))

    # Sample pixels for faster clustering
    sampled_pixels = sample_pixels(pixels)

    # Use MiniBatchKMeans for faster clustering, use random_state = 42 so results are same w/ each run
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)

    #calculate centroids and assign data points to clusters
    labels = kmeans.fit_predict(sampled_pixels)
    centers = kmeans.cluster_centers_.astype("uint8")

    # Map the labels back to the full-resolution image
    full_labels = kmeans.predict(pixels)
    segmented_image = centers[full_labels].reshape(image.shape)

    return segmented_image, centers, full_labels.reshape(image.shape[:2])

#Used to avoid repeat colors
def merge_similar_colors(centers, labels, threshold):
    merged_centers = []
    merged_labels = {}

    # Check distances between cluster centers in the Lab color space
    for i, center in enumerate(centers):
        added = False
        for j, merged_center in enumerate(merged_centers):
            # Calculate the distance in Lab space
            distance = np.linalg.norm(center - merged_center)
            if distance < threshold:
                # Merge colors by assigning the same label
                merged_labels[i] = merged_labels.get(j, j)
                added = True
                break
        if not added:
            merged_centers.append(center)
            merged_labels[i] = len(merged_centers) - 1

    # Update segmented labels to reflect merged colors
    merged_segmented_labels = np.array([merged_labels[label] for label in labels.flatten()])
    merged_segmented_labels = merged_segmented_labels.reshape(labels.shape)

    return merged_centers, merged_segmented_labels

#Matches rgb value of selected area to closest pantone color name using euclidean distance
def match_to_pantone(bgr_color, pantone_colors):
    # Convert BGR to RGB
    rgb_color = tuple(reversed(bgr_color))  # (B, G, R) -> (R, G, B)

    closest_pantone = None
    min_distance = float('inf')

    for name, pantone_rgb in pantone_colors.items():
        # Calculate the Euclidean distance between the input color and Pantone color
        distance = np.linalg.norm(np.array(rgb_color) - np.array(pantone_rgb))

        if distance < min_distance:
            min_distance = distance
            closest_pantone = name

    return closest_pantone

#Generates final image with all colors based on clicked segments
def annotate_segments_with_labels(image, segmented_labels, centers, clicked_points):
    h, w, _ = image.shape
    extended_h = int(h * 1.2)
    extended_w = int(w * 4)
    canvas = np.ones((extended_h, extended_w, 3), dtype="uint8") * 255
    offset_h = (extended_h - h) // 2
    offset_w = (extended_w - w) // 2
    canvas[offset_h:offset_h + h, offset_w:offset_w + w] = image

    label_y = 20
    label_spacing = 30
    label_x = extended_w - 300
    colors_added = []
    # Only annotate colors that were clicked
    for (x, y, color, name) in clicked_points:
        if color not in colors_added:
            # Draw the color on the side pane
            color_text = f"{name}: {color}"
            
            cv2.putText(canvas, color_text, (label_x + 30, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            square_size = 25
            square_y = label_y - 10
            cv2.rectangle(canvas, (label_x, square_y),
                        (label_x + square_size, square_y + square_size), color, -1)
            cv2.rectangle(canvas, (label_x, square_y),
                        (label_x + square_size, square_y + square_size), (0, 0, 0), 1)

            label_y += label_spacing
            colors_added.append(color)

    return canvas

#Gathers data related to user's mouse clicks using segmented areas  
def on_mouse_click(event, x, y, flags, param):
    global clicked_color, clicked_name
    global clicked_points 
    if event == cv2.EVENT_LBUTTONDOWN:
        segmented_labels, centers, image, pantone_colors = param

        # Get the label for the clicked position
        label = segmented_labels[y, x]
        color_lab = centers[label]

        # Convert LAB color to BGR
        color_bgr = cv2.cvtColor(np.uint8([[color_lab]]), cv2.COLOR_Lab2BGR)[0][0]

        # Store the RGB color
        clicked_color = tuple(map(int, color_bgr))  # store RGB directly

        clicked_name = match_to_pantone(color_bgr, pantone_colors)

        # Store the clicked point and color information
        clicked_points.append((x, y, clicked_color, clicked_name))

        # Display the popup with RGB color and Pantone name
        show_color_popup(clicked_color, clicked_name, clicked_color)

#Generates a pop up displaying color info after each click
def show_color_popup(color, name, rgb_value):
  
    # Create an empty image for the popup
    popup = np.ones((100, 300, 3), dtype="uint8") * 255

    # Draw the color box
    cv2.rectangle(popup, (10, 10), (90, 90), color, -1)

    # Display the color name (Pantone)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(popup, f"{name}", (110, 50), font, 0.7, (0, 0, 0), 2)
    cv2.putText(popup, f"{rgb_value}", (110, 80), font, 0.7, (0, 0, 0), 2)

    # Show the popup
    cv2.imshow("Selected Color", popup)

def main():
    global clicked_color, clicked_name
    print("Choose an option:")
    print("1. Identify colors in an image.")
    print("2. Real-time color detection using webcam.")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        image_path = './images/' + input("Enter the path to the image: ")
        image = cv2.imread(image_path)
        if image is None:
            print("Image not found. Please check the path.")
            return

        segmented_image, centers, labels = segment_image_and_sample_colors(
            image, n_clusters=20
        )
        merged_centers, merged_segmented_labels = merge_similar_colors(centers, labels, threshold=30)
       
        cv2.imshow("Image", image)

        # Set the mouse callback to capture clicks
        cv2.setMouseCallback("Image", on_mouse_click, param=(merged_segmented_labels, merged_centers, image, pantone_colors))

        print("\nClick on a color to see its details. Press 'q' to quit and summary will be generated.\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Quit the program
                break
            
        canvas = annotate_segments_with_labels(image, merged_segmented_labels, merged_centers, clicked_points)
        cv2.imshow("Annotated Image", canvas)
        cv2.waitKey(0)  # Wait for a key press to close the annotated image
        cv2.destroyAllWindows()

    elif choice == "2":
        detect_color_from_webcam()

    else:
        print("Invalid choice. Please restart and select 1 or 2.")

if __name__ == "__main__":
    main()
