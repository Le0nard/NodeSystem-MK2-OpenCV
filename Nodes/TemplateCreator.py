import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from NodeEditor import Node, NodePackage, Region, show_region_selector
import os

class TemplateCreator(Node):
    def __init__(self):
        super().__init__("Template Creator", "Analytics", 250)
        self.add_input("image", "image")
        self.add_input("region", "region", default_data=NodePackage())  # Optional region input
        self.add_output("image", "image")
        self.add_output("template", "template")
        self.add_output("region", "region")  # Region output
        
        # UI Controls
        self.x_id = dpg.generate_uuid()
        self.y_id = dpg.generate_uuid()
        self.width_id = dpg.generate_uuid()
        self.height_id = dpg.generate_uuid()
        self.save_id = dpg.generate_uuid()
        self.template_name_id = dpg.generate_uuid()
        self.load_id = dpg.generate_uuid()
        self.templates_combo_id = dpg.generate_uuid()
        
        # Button IDs
        self.full_image_btn_id = dpg.generate_uuid()
        self.select_region_btn_id = dpg.generate_uuid()
        
        # Default values
        self.x = 0
        self.y = 0
        self.width = 100
        self.height = 100
        self.template_name = "template1"
        self.templates_dir = "templates"
        self.templates_list = []
        self.current_template = None
        
        # Store input image for region selection
        self.input_image = None
        self.input_image_shape = None
        
        # Track if region input is connected
        self.region_input_connected = False
        
        # Create templates directory if it doesn't exist
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)
            
        # Load existing templates
        self.update_templates_list()
        
    def update_templates_list(self):
        if os.path.exists(self.templates_dir):
            self.templates_list = [f for f in os.listdir(self.templates_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))]

    def set_full_image(self):
        """Set region parameters to cover the entire input image."""
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
        """Open the region selector popup to visually select template area."""
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
            
            self.update()
        
        show_region_selector(
            image=self.input_image,
            params=params,
            on_confirm=on_region_confirmed
        )

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
        
    def compose(self):
        # Button row for Full Image and Select Region
        with dpg.group(horizontal=True):
            dpg.add_button(label="Full Image", callback=lambda: self.set_full_image(),
                           tag=self.full_image_btn_id, width=115, enabled=False)
            dpg.add_button(label="Select Region", callback=lambda: self.select_region(),
                           tag=self.select_region_btn_id, width=115, enabled=False)
        
        dpg.add_text("Template Region:")
        dpg.add_input_int(
            label="X",
            default_value=self.x,
            callback=self.update_params,
            tag=self.x_id,
            width=185
        )
        dpg.add_input_int(
            label="Y",
            default_value=self.y,
            callback=self.update_params,
            tag=self.y_id,
            width=185
        )
        dpg.add_input_int(
            label="Width",
            default_value=self.width,
            callback=self.update_params,
            tag=self.width_id,
            width=185
        )
        dpg.add_input_int(
            label="Height",
            default_value=self.height,
            callback=self.update_params,
            tag=self.height_id,
            width=185
        )
        
        dpg.add_text("Save Template:")
        dpg.add_input_text(
            label="Name",
            default_value=self.template_name,
            callback=self.update_params,
            tag=self.template_name_id,
            width=185
        )
        dpg.add_button(
            label="Save Template",
            callback=self.save_template,
            tag=self.save_id,
            width=185
        )
        
        dpg.add_text("Load Template:")
        dpg.add_combo(
            items=self.templates_list,
            callback=self.load_template,
            tag=self.templates_combo_id,
            width=185
        )

    def update_params(self):
        if self.region_input_connected:
            return
            
        self.x = max(0, dpg.get_value(self.x_id))
        self.y = max(0, dpg.get_value(self.y_id))
        self.width = max(1, dpg.get_value(self.width_id))
        self.height = max(1, dpg.get_value(self.height_id))
        self.template_name = dpg.get_value(self.template_name_id)
        self.update()
        
    def save_template(self):
        if not self.current_template is None:
            if not self.template_name.endswith(('.png', '.jpg', '.jpeg')):
                self.template_name += '.png'
            
            save_path = os.path.join(self.templates_dir, self.template_name)
            cv2.imwrite(save_path, self.current_template)
            
            # Update templates list
            self.update_templates_list()
            dpg.configure_item(self.templates_combo_id, items=self.templates_list)
            
    def load_template(self):
        selected = dpg.get_value(self.templates_combo_id)
        if selected:
            template_path = os.path.join(self.templates_dir, selected)
            if os.path.exists(template_path):
                self.current_template = cv2.imread(template_path)
                self.update()

    def execute(self, inputs: list[NodePackage]) -> list[NodePackage]:
        # Get inputs
        image_data = inputs[0] if len(inputs) > 0 else None
        region_data = inputs[1] if len(inputs) > 1 else None
        
        image = image_data.image_or_mask if image_data else None
        
        if image is None:
            self.input_image = None
            self.input_image_shape = None
            self.region_input_connected = False
            self._set_controls_enabled(False)
            return [NodePackage(), NodePackage(), NodePackage()]
        
        h, w = image.shape[:2]
        
        # Store image for region selection
        self.input_image = image.copy()
        self.input_image_shape = image.shape
        
        # Check if region input is connected
        if region_data and region_data.region is not None:
            self.region_input_connected = True
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
            
            self._set_controls_enabled(False)
        else:
            self.region_input_connected = False
            self._set_controls_enabled(True)
        
        # Constrain selection to image bounds
        self.x = min(max(0, self.x), w - 1)
        self.y = min(max(0, self.y), h - 1)
        self.width = min(self.width, w - self.x)
        self.height = min(self.height, h - self.y)
        
        # Create visualization
        vis_image = image.copy()
        cv2.rectangle(vis_image, 
                     (self.x, self.y), 
                     (self.x + self.width, self.y + self.height),
                     (0, 255, 0), 2)
        
        # Extract template region
        self.current_template = image[self.y:self.y + self.height,
                                    self.x:self.x + self.width].copy()
        
        # Create output region
        output_region = Region(x=self.x, y=self.y, width=self.width, height=self.height)
        
        return [
            NodePackage(image_or_mask=vis_image),
            NodePackage(image_or_mask=self.current_template),
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