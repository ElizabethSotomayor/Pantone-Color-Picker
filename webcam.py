import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from colors import pantone_colors

#Opens a webcam feed and detects colors in real-time when the user clicks on the video frame
def detect_color_from_webcam():
  
    detected_colors = []  # List to store all detected colors

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

    #Callback function to handle mouse clicks on the video frame.
    def mouse_callback(event, x, y, flags, param):
      
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr_color = frame[y, x]

            color_name = match_to_pantone(bgr_color, pantone_colors)
            detected_colors.append((x, y, color_name, bgr_color))
            cv2.putText(
                frame,
                f"{color_name}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            show_color_popup(bgr_color, color_name, bgr_color)

    #Generates a popup window with color info after each click
    def show_color_popup(color, name, rgb_value):
       
        # Create an empty image for the popup
        popup = np.ones((100, 300, 3), dtype="uint8") * 255
        color = tuple(int(c) for c in color) 
        # Draw the color box
        cv2.rectangle(popup, (10, 10), (90, 90), color, -1)

        # Display the color name
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(popup, name, (110, 50), font, 0.7, (0, 0, 0), 2)
        formatted_rgb = f"({rgb_value[2]}, {rgb_value[1]}, {rgb_value[0]})"
        cv2.putText(popup, formatted_rgb, (110, 80), font, 0.7, (0, 0, 0), 2)

        # Show the popup
        cv2.imshow("Selected Color", popup)

    #Generates an image showing summary of colors clicked 
    def create_color_squares_image():
       
        square_size = 70
        margin = 20
        width = (square_size + margin) * len(detected_colors) + margin
        height = square_size + margin * 2

        # Create a blank white image
        output_image = np.ones((height + 50, width + 50, 3), dtype=np.uint8) * 255

        for i, (x, y, color_name, bgr) in enumerate(detected_colors):
            # Calculate position of the square
            start_x = margin + i * (square_size + margin)
            end_x = start_x + square_size
            start_y = margin
            end_y = start_y + square_size

            # Draw the color square
            output_image[start_y:end_y, start_x:end_x] = bgr

            rgb_value = tuple(reversed(bgr))  # Convert from BGR to RGB
            formatted_rgb = f"({rgb_value[0]}, {rgb_value[1]}, {rgb_value[2]})"  # Format as (r, g, b)
            # Add the color name 
            label = f"{color_name}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = start_x + (square_size - text_size[0]) // 2
            text_y = end_y + 15
            cv2.putText(
            output_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            )   

            rgb_value = (bgr[2], bgr[1], bgr[0])  # Convert BGR to RGB
            rgb_text = f"({rgb_value[0]}, {rgb_value[1]}, {rgb_value[2]})"
            
            # Display the RGB value beneath the color name
            rgb_text_size = cv2.getTextSize(rgb_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            rgb_x = start_x + (square_size - rgb_text_size[0]) // 2
            rgb_y = text_y + 15  # Adjust y position for the RGB value, below the label
            cv2.putText(
                output_image,
                rgb_text,
                (rgb_x, rgb_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
        

        return output_image

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Click on the video feed to detect colors.")
    print("Press 'q' to quit and show colors found or 'c' to clear detected colors.\n")

    cv2.namedWindow("Webcam Feed")
    cv2.setMouseCallback("Webcam Feed", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam.")
            break

        # Show the video feed
        cv2.imshow("Webcam Feed", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Clear detected colors
            detected_colors.clear()
            print("Detected colors cleared.\n")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Create and save the color squares image
    if detected_colors:
        color_squares_image = create_color_squares_image()
        output_filename = "detected_colors.png"
        cv2.imwrite(output_filename, color_squares_image)
        print(f"Color squares image saved as {output_filename}.")
        cv2.imshow("Detected Colors", color_squares_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nNo colors were detected.")
