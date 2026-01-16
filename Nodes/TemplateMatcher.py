import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from NodeEditor import Node, NodePackage, Region, show_region_selector

class TemplateMatcher(Node):
    def __init__(self):
        super().__init__("Template Matcher", "Analytics", 250)
        self.add_input("image", "image")
        self.add_input("template", "template")
        self.add_input("region", "region", default_data=NodePackage())  # Optional region input for search area
        self.add_output("image", "image")  # Visualization
        self.add_output("mask", "mask")
        self.add_output("region", "region")  # Region output (first match)
        
        # UI Controls
        self.method_id = dpg.generate_uuid()
        self.threshold_id = dpg.generate_uuid()
        self.max_matches_id = dpg.generate_uuid()
        
        # Search region controls
        self.x_id = dpg.generate_uuid()
        self.y_id = dpg.generate_uuid()
        self.width_id = dpg.generate_uuid()
        self.height_id = dpg.generate_uuid()
        self.use_region_id = dpg.generate_uuid()
        
        # Button IDs
        self.full_image_btn_id = dpg.generate_uuid()
        self.select_region_btn_id = dpg.generate_uuid()
        
        # Default values
        self.method = cv2.TM_CCOEFF_NORMED
        self.threshold = 0.8
        self.max_matches = 3
        
        # Search region
        self.x = 0
        self.y = 0
        self.width = 100
        self.height = 100
        self.use_region = False
        
        # Store input image for region selection
        self.input_image = None
        self.input_image_shape = None
        
        # Track if region input is connected
        self.region_input_connected = False
        
    def set_full_image(self):
        """Set search region to cover the entire input image."""
        if self.input_image_shape is not None and not self.region_input_connected:
            img_height, img_width = self.input_image_shape[:2]
            self.x = 0
            self.y = 0
            self.width = img_width
            self.height = img_height
            
            dpg.set_value(self.x_id, self.x)
            dpg.set_value(self.y_id, self.y)
            dpg.set_value(self.width_id, self.width)
            dpg.set_value(self.height_id, self.height)
            
            self.update()

    def select_region(self):
        """Open the region selector popup to visually select search area."""
        if self.input_image is None or self.region_input_connected:
            return
        
        params = {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }
        
        def on_region_confirmed(new_params):
            self.x = new_params['x']
            self.y = new_params['y']
            self.width = new_params['width']
            self.height = new_params['height']
            
            dpg.set_value(self.x_id, self.x)
            dpg.set_value(self.y_id, self.y)
            dpg.set_value(self.width_id, self.width)
            dpg.set_value(self.height_id, self.height)
            
            # Enable use_region since user selected one
            self.use_region = True
            dpg.set_value(self.use_region_id, True)
            
            self.update()
        
        show_region_selector(
            image=self.input_image,
            params=params,
            on_confirm=on_region_confirmed
        )

    def _set_region_controls_enabled(self, enabled: bool):
        """Enable or disable region parameter controls."""
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
        if dpg.does_item_exist(self.use_region_id):
            dpg.configure_item(self.use_region_id, enabled=enabled)
        
    def compose(self):
        dpg.add_text("Match Method:")
        dpg.add_combo(
            items=["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"],
            default_value="TM_CCOEFF_NORMED",
            callback=self.update_params,
            tag=self.method_id,
            width=185
        )
        
        dpg.add_text("Threshold:")
        dpg.add_slider_float(
            default_value=self.threshold,
            min_value=0.1,
            max_value=1.0,
            callback=self.update_params,
            tag=self.threshold_id,
            width=185
        )
        
        dpg.add_text("Max Matches:")
        dpg.add_input_int(
            default_value=self.max_matches,
            min_value=1,
            max_value=20,
            callback=self.update_params,
            tag=self.max_matches_id,
            width=185
        )
        
        dpg.add_spacer(height=5)
        dpg.add_checkbox(label="Use Search Region", default_value=self.use_region,
                        callback=self.update_params, tag=self.use_region_id)
        
        # Button row for Full Image and Select Region
        with dpg.group(horizontal=True):
            dpg.add_button(label="Full Image", callback=lambda: self.set_full_image(),
                           tag=self.full_image_btn_id, width=115, enabled=False)
            dpg.add_button(label="Select Region", callback=lambda: self.select_region(),
                           tag=self.select_region_btn_id, width=115, enabled=False)
        
        dpg.add_text("Search Region:")
        dpg.add_input_int(label="X", default_value=self.x,
                         callback=self.update_params, tag=self.x_id, width=185)
        dpg.add_input_int(label="Y", default_value=self.y,
                         callback=self.update_params, tag=self.y_id, width=185)
        dpg.add_input_int(label="Width", default_value=self.width,
                         callback=self.update_params, tag=self.width_id, width=185)
        dpg.add_input_int(label="Height", default_value=self.height,
                         callback=self.update_params, tag=self.height_id, width=185)

    def update_params(self):
        method_text = dpg.get_value(self.method_id)
        self.method = {
            "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
        }.get(method_text, cv2.TM_CCOEFF_NORMED)
        
        self.threshold = dpg.get_value(self.threshold_id)
        self.max_matches = dpg.get_value(self.max_matches_id)
        self.use_region = dpg.get_value(self.use_region_id)
        
        if not self.region_input_connected:
            self.x = max(0, dpg.get_value(self.x_id))
            self.y = max(0, dpg.get_value(self.y_id))
            self.width = max(1, dpg.get_value(self.width_id))
            self.height = max(1, dpg.get_value(self.height_id))
        
        self.update()

    def execute(self, inputs: list[NodePackage]) -> list[NodePackage]:
        # Get inputs
        image_data = inputs[0] if len(inputs) > 0 else None
        template_data = inputs[1] if len(inputs) > 1 else None
        region_data = inputs[2] if len(inputs) > 2 else None
        
        image = image_data.image_or_mask if image_data else None
        template = template_data.image_or_mask if template_data else None
        
        if image is None or template is None:
            self.input_image = None
            self.input_image_shape = None
            self.region_input_connected = False
            self._set_region_controls_enabled(False)
            return [NodePackage(), NodePackage(), NodePackage()]
        
        # Store image for region selection
        self.input_image = image.copy()
        self.input_image_shape = image.shape
        
        # Check if region input is connected
        if region_data and region_data.region is not None:
            self.region_input_connected = True
            self.use_region = True
            input_region = region_data.region
            self.x = input_region.x
            self.y = input_region.y
            self.width = input_region.width
            self.height = input_region.height
            
            # Update UI values
            if dpg.does_item_exist(self.x_id):
                dpg.set_value(self.x_id, self.x)
            if dpg.does_item_exist(self.y_id):
                dpg.set_value(self.y_id, self.y)
            if dpg.does_item_exist(self.width_id):
                dpg.set_value(self.width_id, self.width)
            if dpg.does_item_exist(self.height_id):
                dpg.set_value(self.height_id, self.height)
            if dpg.does_item_exist(self.use_region_id):
                dpg.set_value(self.use_region_id, True)
            
            self._set_region_controls_enabled(False)
        else:
            self.region_input_connected = False
            self._set_region_controls_enabled(True)
        
        # Determine search area
        if self.use_region:
            h, w = image.shape[:2]
            sx = min(max(0, self.x), w - 1)
            sy = min(max(0, self.y), h - 1)
            sw = min(self.width, w - sx)
            sh = min(self.height, h - sy)
            search_image = image[sy:sy+sh, sx:sx+sw]
            offset_x, offset_y = sx, sy
        else:
            search_image = image
            offset_x, offset_y = 0, 0
            
        # Convert both to grayscale if needed
        if len(search_image.shape) == 3:
            gray_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = search_image.copy()
            
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template.copy()
            
        # Match template
        result = cv2.matchTemplate(gray_image, gray_template, self.method)
        
        # Find matches
        if self.method == cv2.TM_SQDIFF_NORMED:
            matches = np.where(result <= 1.0 - self.threshold)
        else:
            matches = np.where(result >= self.threshold)
            
        # Create visualization and mask
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
            
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        th, tw = template.shape[:2]
        
        # Draw search region if used
        if self.use_region:
            cv2.rectangle(vis_image, 
                         (self.x, self.y), 
                         (self.x + self.width, self.y + self.height),
                         (255, 255, 0), 1)
        
        # Draw rectangles around matches
        matches_list = list(zip(*matches[::-1]))
        matches_list = sorted(matches_list, key=lambda x: result[x[1], x[0]], reverse=True)
        matches_list = matches_list[:self.max_matches]
        
        first_match_region = None
        for i, pt in enumerate(matches_list):
            # Adjust for search region offset
            abs_x = pt[0] + offset_x
            abs_y = pt[1] + offset_y
            cv2.rectangle(vis_image, (abs_x, abs_y), (abs_x + tw, abs_y + th), (0, 255, 0), 2)
            cv2.rectangle(mask, (abs_x, abs_y), (abs_x + tw, abs_y + th), 255, -1)
            
            # Store first match as output region
            if i == 0:
                first_match_region = Region(x=abs_x, y=abs_y, width=tw, height=th)

        return [
            NodePackage(image_or_mask=vis_image),
            NodePackage(image_or_mask=mask),
            NodePackage(region=first_match_region)
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