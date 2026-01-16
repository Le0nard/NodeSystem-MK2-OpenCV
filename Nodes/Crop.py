import cv2
import numpy as np
from NodeEditor import Node, NodePackage, dpg

class Crop(Node):
    def __init__(self):
        super().__init__("Crop", "Operations", 200)
        self.add_input("image")
        self.add_output("image")
        
        # UI Controls
        self.x_id = dpg.generate_uuid()
        self.y_id = dpg.generate_uuid()
        self.width_id = dpg.generate_uuid()
        self.height_id = dpg.generate_uuid()
        self.maintain_aspect_id = dpg.generate_uuid()
        
        # Default values
        self.x = 0
        self.y = 0
        self.width = 100
        self.height = 100
        self.maintain_aspect = True
        self.aspect_ratio = 1.0
        
        # Full Image button
        self.full_image_btn_id = dpg.generate_uuid()
        self.input_image_shape = None  # Store input image dimensions

    def set_full_image(self):
        """Set crop parameters to cover the entire input image."""
        if self.input_image_shape is not None:
            img_height, img_width = self.input_image_shape[:2]
            self.x = 0
            self.y = 0
            self.width = img_width
            self.height = img_height
            
            # Update UI controls
            dpg.set_value(self.x_id, self.x)
            dpg.set_value(self.y_id, self.y)
            dpg.set_value(self.width_id, self.width)
            dpg.set_value(self.height_id, self.height)
            
            self.update()

    def on_save(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "maintain_aspect": self.maintain_aspect
        }
    
    def on_load(self, data: dict):
        self.x = data["x"]
        self.y = data["y"]
        self.width = data["width"]
        self.height = data["height"]
        self.maintain_aspect = data["maintain_aspect"]
        self.update()

    def update_params(self):
        self.x = dpg.get_value(self.x_id)
        self.y = dpg.get_value(self.y_id)
        self.width = dpg.get_value(self.width_id)
        self.height = dpg.get_value(self.height_id)
        self.maintain_aspect = dpg.get_value(self.maintain_aspect_id)
        
        if self.maintain_aspect and self.aspect_ratio > 0:
            # Adjust height to maintain aspect ratio when width changes
            self.height = int(self.width / self.aspect_ratio)
            dpg.set_value(self.height_id, self.height)
            
        self.update()

    def compose(self):
        # Full Image button - disabled by default until an image is provided
        dpg.add_button(label="Full Image", callback=lambda: self.set_full_image(),
                       tag=self.full_image_btn_id, width=200, enabled=False)
        dpg.add_text("Crop Parameters:")
        dpg.add_input_int(label="X", default_value=self.x,
                         callback=self.update_params, tag=self.x_id, width=185)
        dpg.add_input_int(label="Y", default_value=self.y,
                         callback=self.update_params, tag=self.y_id, width=185)
        dpg.add_input_int(label="Width", default_value=self.width,
                         callback=self.update_params, tag=self.width_id, width=185)
        dpg.add_input_int(label="Height", default_value=self.height,
                         callback=self.update_params, tag=self.height_id, width=185)
        dpg.add_checkbox(label="Maintain Aspect Ratio", default_value=self.maintain_aspect,
                        callback=self.update_params, tag=self.maintain_aspect_id)

    def execute(self, inputs: list[NodePackage]) -> list[NodePackage]:
        data = inputs[0]
        image = data.image_or_mask
        
        if image is None:
            # Disable the Full Image button when no image is available
            self.input_image_shape = None
            if dpg.does_item_exist(self.full_image_btn_id):
                dpg.configure_item(self.full_image_btn_id, enabled=False)
            return [NodePackage()]
        
        # Store image dimensions and enable the Full Image button
        self.input_image_shape = image.shape
        if dpg.does_item_exist(self.full_image_btn_id):
            dpg.configure_item(self.full_image_btn_id, enabled=True)
            
        # Update aspect ratio based on input image
        if self.maintain_aspect:
            self.aspect_ratio = image.shape[1] / image.shape[0]
        
        # Ensure crop region is within image bounds
        x = max(0, min(self.x, image.shape[1]))
        y = max(0, min(self.y, image.shape[0]))
        width = min(self.width, image.shape[1] - x)
        height = min(self.height, image.shape[0] - y)
        
        # Perform crop
        cropped = image[y:y+height, x:x+width]
        
        return [NodePackage(image_or_mask=cropped)]

    def viewer(self, outputs: list[NodePackage]):
        data = outputs[0]
        img_tag = dpg.generate_uuid()
        with dpg.texture_registry():
            dpg.add_dynamic_texture(400, 400, [0.0, 0.0, 0.0, 0.0]*400*400, tag=img_tag)
        
        dpg.add_image(img_tag)
        
        image_rgba = data.copy_resize((400, 400), keep_alpha=True)
        image_rgba = image_rgba.astype(float)
        image_rgba /= 255

        dpg.set_value(img_tag, image_rgba.flatten())