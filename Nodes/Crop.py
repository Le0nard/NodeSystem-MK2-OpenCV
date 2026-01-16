import cv2
import numpy as np
from NodeEditor import Node, NodePackage, Region, dpg, show_region_selector

class Crop(Node):
    def __init__(self):
        super().__init__("Crop", "Operations", 200)
        self.add_input("image", "image")
        self.add_input("region", "region", default_data=NodePackage())  # Optional region input
        self.add_output("image", "image")
        self.add_output("region", "region")  # Region output
        
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
        
        # Select Region button
        self.select_region_btn_id = dpg.generate_uuid()
        self.input_image = None  # Store reference to input image for region selection
        
        # Track if region input is connected
        self.region_input_connected = False

    def set_full_image(self):
        """Set crop parameters to cover the entire input image."""
        if self.input_image_shape is not None and not self.region_input_connected:
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

    def select_region(self):
        """Open the region selector popup to visually select crop area."""
        if self.input_image is None or self.region_input_connected:
            return
        
        # Current region parameters
        params = {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }
        
        def on_region_confirmed(new_params):
            """Callback when OK is clicked in the region selector."""
            self.x = new_params['x']
            self.y = new_params['y']
            self.width = new_params['width']
            self.height = new_params['height']
            
            # Update UI controls
            dpg.set_value(self.x_id, self.x)
            dpg.set_value(self.y_id, self.y)
            dpg.set_value(self.width_id, self.width)
            dpg.set_value(self.height_id, self.height)
            
            self.update()
        
        # Show region selector popup
        show_region_selector(
            image=self.input_image,
            params=params,
            on_confirm=on_region_confirmed
        )

    def on_save(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "maintain_aspect": self.maintain_aspect
        }
    
    def on_load(self, data: dict):
        self.x = data.get("x", 0)
        self.y = data.get("y", 0)
        self.width = data.get("width", 100)
        self.height = data.get("height", 100)
        self.maintain_aspect = data.get("maintain_aspect", True)
        self.update()

    def update_params(self):
        if self.region_input_connected:
            return  # Don't update params when region input is connected
            
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

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all parameter controls."""
        if dpg.does_item_exist(self.full_image_btn_id):
            dpg.configure_item(self.full_image_btn_id, enabled=enabled)
        if dpg.does_item_exist(self.select_region_btn_id):
            dpg.configure_item(self.select_region_btn_id, enabled=enabled)
        if dpg.does_item_exist(self.x_id):
            dpg.configure_item(self.x_id, enabled=enabled)
        if dpg.does_item_exist(self.y_id):
            dpg.configure_item(self.y_id, enabled=enabled)
        if dpg.does_item_exist(self.width_id):
            dpg.configure_item(self.width_id, enabled=enabled)
        if dpg.does_item_exist(self.height_id):
            dpg.configure_item(self.height_id, enabled=enabled)
        if dpg.does_item_exist(self.maintain_aspect_id):
            dpg.configure_item(self.maintain_aspect_id, enabled=enabled)

    def compose(self):
        # Button row for Full Image and Select Region
        with dpg.group(horizontal=True):
            dpg.add_button(label="Full Image", callback=lambda: self.set_full_image(),
                           tag=self.full_image_btn_id, width=95, enabled=False)
            dpg.add_button(label="Select Region", callback=lambda: self.select_region(),
                           tag=self.select_region_btn_id, width=95, enabled=False)
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
        # Get image input
        image_data = inputs[0] if len(inputs) > 0 else None
        region_data = inputs[1] if len(inputs) > 1 else None
        
        image = image_data.image_or_mask if image_data else None
        
        if image is None:
            # Disable buttons when no image is available
            self.input_image_shape = None
            self.input_image = None
            self.region_input_connected = False
            self._set_controls_enabled(False)
            return [NodePackage(), NodePackage()]
        
        # Store image and dimensions
        self.input_image_shape = image.shape
        self.input_image = image.copy()
        
        # Check if region input is connected and has valid data
        if region_data and region_data.region is not None:
            self.region_input_connected = True
            # Use region input values
            input_region = region_data.region
            self.x = input_region.x
            self.y = input_region.y
            self.width = input_region.width
            self.height = input_region.height
            
            # Update UI to show current values (but keep disabled)
            if dpg.does_item_exist(self.x_id):
                dpg.set_value(self.x_id, self.x)
            if dpg.does_item_exist(self.y_id):
                dpg.set_value(self.y_id, self.y)
            if dpg.does_item_exist(self.width_id):
                dpg.set_value(self.width_id, self.width)
            if dpg.does_item_exist(self.height_id):
                dpg.set_value(self.height_id, self.height)
            
            # Disable controls when region input is connected
            self._set_controls_enabled(False)
        else:
            self.region_input_connected = False
            # Enable controls when no region input
            self._set_controls_enabled(True)
            
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
        
        # Create output region
        output_region = Region(x=x, y=y, width=width, height=height)
        
        return [
            NodePackage(image_or_mask=cropped),
            NodePackage(region=output_region)
        ]

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