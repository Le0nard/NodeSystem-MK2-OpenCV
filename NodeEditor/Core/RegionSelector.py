"""
RegionSelector - A reusable module for visual region selection in VisionNode.

This module provides a popup window that allows users to interactively
select a rectangular region on an image by dragging and resizing.
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Callable, Optional, Dict, Any


class RegionSelector:
    """
    A class that provides visual region selection functionality.
    
    Usage:
        selector = RegionSelector()
        selector.show(
            image=numpy_image,
            params={'x': 0, 'y': 0, 'width': 100, 'height': 100},
            on_confirm=lambda p: print(f"Selected: {p}"),
            on_cancel=lambda: print("Cancelled")
        )
    """
    
    def __init__(self):
        self._window_id = None
        self._drawlist_id = None
        self._texture_id = None
        self._image_id = None
        
        # Region parameters
        self._region_x = 0
        self._region_y = 0
        self._region_width = 100
        self._region_height = 100
        
        # Image data
        self._image = None
        self._image_height = 0
        self._image_width = 0
        self._display_scale = 1.0
        
        # Callbacks
        self._on_confirm: Optional[Callable[[Dict[str, int]], None]] = None
        self._on_cancel: Optional[Callable[[], None]] = None
        
        # Drag state
        self._is_dragging = False
        self._is_resizing = False
        self._resize_handle = None  # 'tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r'
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_region_x = 0
        self._drag_start_region_y = 0
        self._drag_start_region_w = 0
        self._drag_start_region_h = 0
        
        # Handler registry
        self._handler_registry_id = None
        
        # Offset for image position within window
        self._image_offset_x = 10
        self._image_offset_y = 40  # Account for title bar
        
    def show(self, image: np.ndarray, params: Dict[str, int],
             on_confirm: Callable[[Dict[str, int]], None],
             on_cancel: Optional[Callable[[], None]] = None,
             max_display_size: int = 800):
        """
        Show the region selector popup.
        
        Args:
            image: The input image as a numpy array (BGR or BGRA format from OpenCV)
            params: Dictionary with 'x', 'y', 'width', 'height' keys
            on_confirm: Callback function called with new params when OK is clicked
            on_cancel: Optional callback function called when Cancel is clicked
            max_display_size: Maximum display size for the image (will scale if larger)
        """
        self._image = image
        self._image_height, self._image_width = image.shape[:2]
        
        # Load initial region parameters
        self._region_x = params.get('x', 0)
        self._region_y = params.get('y', 0)
        self._region_width = params.get('width', 100)
        self._region_height = params.get('height', 100)
        
        # Store callbacks
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel
        
        # Calculate display scale to fit within max_display_size
        scale_w = max_display_size / self._image_width if self._image_width > max_display_size else 1.0
        scale_h = max_display_size / self._image_height if self._image_height > max_display_size else 1.0
        self._display_scale = min(scale_w, scale_h)
        
        display_width = int(self._image_width * self._display_scale)
        display_height = int(self._image_height * self._display_scale)
        
        # Create unique IDs
        self._window_id = dpg.generate_uuid()
        self._drawlist_id = dpg.generate_uuid()
        self._texture_id = dpg.generate_uuid()
        self._handler_registry_id = dpg.generate_uuid()
        
        # Prepare image for display (convert to RGBA float format for DearPyGui)
        image_rgba = self._prepare_image_for_display(image, display_width, display_height)
        
        # Create texture
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                display_width, display_height,
                image_rgba.flatten().tolist(),
                tag=self._texture_id
            )
        
        # Create window
        window_width = display_width + 20
        window_height = display_height + 100  # Extra space for buttons
        
        with dpg.window(
            label="Select Region",
            modal=True,
            tag=self._window_id,
            width=window_width,
            height=window_height,
            no_resize=True,
            on_close=self._on_close_window
        ):
            # Image with drawlist overlay
            with dpg.drawlist(
                width=display_width,
                height=display_height,
                tag=self._drawlist_id
            ):
                # Draw image
                dpg.draw_image(
                    self._texture_id,
                    pmin=[0, 0],
                    pmax=[display_width, display_height]
                )
                
            # Draw initial region
            self._update_region_drawing()
            
            dpg.add_spacer(height=10)
            
            # Buttons row
            with dpg.group(horizontal=True):
                dpg.add_button(label="Cancel", callback=self._on_cancel_click, width=100)
                dpg.add_spacer(width=window_width - 220)
                dpg.add_button(label="OK", callback=self._on_ok_click, width=100)
        
        # Set up mouse handlers for this window
        with dpg.handler_registry(tag=self._handler_registry_id):
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_click)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self._on_mouse_release)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
    
    def _prepare_image_for_display(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Prepare image for DearPyGui display (resize and convert to RGBA float)."""
        import cv2
        
        # Resize image
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        
        # Convert to RGBA if needed
        if len(resized.shape) == 2:
            # Grayscale
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGBA)
        elif resized.shape[2] == 3:
            # BGR to RGBA
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        elif resized.shape[2] == 4:
            # BGRA to RGBA
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA)
        
        # Convert to float [0, 1]
        return resized.astype(np.float32) / 255.0
    
    def _update_region_drawing(self):
        """Update the region rectangle on the drawlist."""
        if not dpg.does_item_exist(self._drawlist_id):
            return
            
        # Delete existing region drawings (keep the image)
        children = dpg.get_item_children(self._drawlist_id, slot=2)
        if children:
            for child in children[1:]:  # Skip first child (the image)
                if dpg.does_item_exist(child):
                    dpg.delete_item(child)
        
        # Calculate scaled region coordinates
        x1 = int(self._region_x * self._display_scale)
        y1 = int(self._region_y * self._display_scale)
        x2 = int((self._region_x + self._region_width) * self._display_scale)
        y2 = int((self._region_y + self._region_height) * self._display_scale)
        
        # Draw region rectangle (red, semi-transparent fill)
        dpg.draw_rectangle(
            pmin=[x1, y1],
            pmax=[x2, y2],
            color=[255, 0, 0, 255],
            fill=[255, 0, 0, 50],
            thickness=2,
            parent=self._drawlist_id
        )
        
        # Draw resize handles (small squares at corners and edges)
        handle_size = 8
        handle_color = [255, 255, 255, 255]
        handle_fill = [255, 0, 0, 255]
        
        # Corner handles
        handles = [
            (x1, y1),  # top-left
            (x2, y1),  # top-right
            (x1, y2),  # bottom-left
            (x2, y2),  # bottom-right
        ]
        
        for hx, hy in handles:
            dpg.draw_rectangle(
                pmin=[hx - handle_size//2, hy - handle_size//2],
                pmax=[hx + handle_size//2, hy + handle_size//2],
                color=handle_color,
                fill=handle_fill,
                thickness=1,
                parent=self._drawlist_id
            )
        
        # Edge handles (center of each edge)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        edge_handles = [
            (mid_x, y1),  # top
            (mid_x, y2),  # bottom
            (x1, mid_y),  # left
            (x2, mid_y),  # right
        ]
        
        for hx, hy in edge_handles:
            dpg.draw_rectangle(
                pmin=[hx - handle_size//2, hy - handle_size//2],
                pmax=[hx + handle_size//2, hy + handle_size//2],
                color=handle_color,
                fill=handle_fill,
                thickness=1,
                parent=self._drawlist_id
            )
    
    def _get_mouse_pos_in_image(self) -> tuple:
        """Get mouse position relative to the image in the drawlist."""
        mouse_pos = dpg.get_mouse_pos(local=False)
        window_pos = dpg.get_item_pos(self._window_id)
        
        # Calculate position relative to drawlist
        rel_x = mouse_pos[0] - window_pos[0] - self._image_offset_x
        rel_y = mouse_pos[1] - window_pos[1] - self._image_offset_y
        
        return rel_x, rel_y
    
    def _get_resize_handle_at(self, rel_x: float, rel_y: float) -> Optional[str]:
        """Check if mouse is over a resize handle and return which one."""
        handle_size = 12  # Slightly larger hit area
        
        # Scaled region coordinates
        x1 = int(self._region_x * self._display_scale)
        y1 = int(self._region_y * self._display_scale)
        x2 = int((self._region_x + self._region_width) * self._display_scale)
        y2 = int((self._region_y + self._region_height) * self._display_scale)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        # Check corners first
        if abs(rel_x - x1) < handle_size and abs(rel_y - y1) < handle_size:
            return 'tl'
        if abs(rel_x - x2) < handle_size and abs(rel_y - y1) < handle_size:
            return 'tr'
        if abs(rel_x - x1) < handle_size and abs(rel_y - y2) < handle_size:
            return 'bl'
        if abs(rel_x - x2) < handle_size and abs(rel_y - y2) < handle_size:
            return 'br'
        
        # Check edges
        if abs(rel_x - mid_x) < handle_size and abs(rel_y - y1) < handle_size:
            return 't'
        if abs(rel_x - mid_x) < handle_size and abs(rel_y - y2) < handle_size:
            return 'b'
        if abs(rel_x - x1) < handle_size and abs(rel_y - mid_y) < handle_size:
            return 'l'
        if abs(rel_x - x2) < handle_size and abs(rel_y - mid_y) < handle_size:
            return 'r'
        
        return None
    
    def _is_inside_region(self, rel_x: float, rel_y: float) -> bool:
        """Check if mouse is inside the region (but not on a handle)."""
        x1 = int(self._region_x * self._display_scale)
        y1 = int(self._region_y * self._display_scale)
        x2 = int((self._region_x + self._region_width) * self._display_scale)
        y2 = int((self._region_y + self._region_height) * self._display_scale)
        
        return x1 < rel_x < x2 and y1 < rel_y < y2
    
    def _on_mouse_click(self, sender, app_data):
        """Handle mouse click for starting drag or resize."""
        if not dpg.does_item_exist(self._window_id):
            return
            
        rel_x, rel_y = self._get_mouse_pos_in_image()
        
        # Check for resize handle
        handle = self._get_resize_handle_at(rel_x, rel_y)
        if handle:
            self._is_resizing = True
            self._resize_handle = handle
            self._drag_start_x = rel_x
            self._drag_start_y = rel_y
            self._drag_start_region_x = self._region_x
            self._drag_start_region_y = self._region_y
            self._drag_start_region_w = self._region_width
            self._drag_start_region_h = self._region_height
            return
        
        # Check for drag inside region
        if self._is_inside_region(rel_x, rel_y):
            self._is_dragging = True
            self._drag_start_x = rel_x
            self._drag_start_y = rel_y
            self._drag_start_region_x = self._region_x
            self._drag_start_region_y = self._region_y
    
    def _on_mouse_release(self, sender, app_data):
        """Handle mouse release to end drag or resize."""
        self._is_dragging = False
        self._is_resizing = False
        self._resize_handle = None
    
    def _on_mouse_move(self, sender, app_data):
        """Handle mouse move for dragging or resizing."""
        if not dpg.does_item_exist(self._window_id):
            return
            
        rel_x, rel_y = self._get_mouse_pos_in_image()
        
        if self._is_dragging:
            # Calculate delta in image coordinates
            delta_x = (rel_x - self._drag_start_x) / self._display_scale
            delta_y = (rel_y - self._drag_start_y) / self._display_scale
            
            # Update region position
            new_x = self._drag_start_region_x + delta_x
            new_y = self._drag_start_region_y + delta_y
            
            # Clamp to image bounds
            new_x = max(0, min(new_x, self._image_width - self._region_width))
            new_y = max(0, min(new_y, self._image_height - self._region_height))
            
            self._region_x = int(new_x)
            self._region_y = int(new_y)
            
            self._update_region_drawing()
            
        elif self._is_resizing and self._resize_handle:
            # Calculate delta in image coordinates
            delta_x = (rel_x - self._drag_start_x) / self._display_scale
            delta_y = (rel_y - self._drag_start_y) / self._display_scale
            
            new_x = self._drag_start_region_x
            new_y = self._drag_start_region_y
            new_w = self._drag_start_region_w
            new_h = self._drag_start_region_h
            
            min_size = 10  # Minimum region size
            
            # Apply resize based on handle
            if 'l' in self._resize_handle:
                new_x = self._drag_start_region_x + delta_x
                new_w = self._drag_start_region_w - delta_x
            if 'r' in self._resize_handle:
                new_w = self._drag_start_region_w + delta_x
            if 't' in self._resize_handle:
                new_y = self._drag_start_region_y + delta_y
                new_h = self._drag_start_region_h - delta_y
            if 'b' in self._resize_handle:
                new_h = self._drag_start_region_h + delta_y
            
            # Clamp values
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_w = max(min_size, min(new_w, self._image_width - new_x))
            new_h = max(min_size, min(new_h, self._image_height - new_y))
            
            self._region_x = int(new_x)
            self._region_y = int(new_y)
            self._region_width = int(new_w)
            self._region_height = int(new_h)
            
            self._update_region_drawing()
    
    def _on_ok_click(self, sender, app_data):
        """Handle OK button click."""
        params = {
            'x': self._region_x,
            'y': self._region_y,
            'width': self._region_width,
            'height': self._region_height
        }
        
        self._cleanup()
        
        if self._on_confirm:
            self._on_confirm(params)
    
    def _on_cancel_click(self, sender, app_data):
        """Handle Cancel button click."""
        self._cleanup()
        
        if self._on_cancel:
            self._on_cancel()
    
    def _on_close_window(self, sender, app_data):
        """Handle window close (X button)."""
        self._cleanup()
        
        if self._on_cancel:
            self._on_cancel()
    
    def _cleanup(self):
        """Clean up resources."""
        # Delete handler registry
        if self._handler_registry_id and dpg.does_item_exist(self._handler_registry_id):
            dpg.delete_item(self._handler_registry_id)
        
        # Delete texture
        if self._texture_id and dpg.does_item_exist(self._texture_id):
            dpg.delete_item(self._texture_id)
        
        # Delete window
        if self._window_id and dpg.does_item_exist(self._window_id):
            dpg.delete_item(self._window_id)


def show_region_selector(image: np.ndarray, params: Dict[str, int],
                          on_confirm: Callable[[Dict[str, int]], None],
                          on_cancel: Optional[Callable[[], None]] = None,
                          max_display_size: int = 800):
    """
    Convenience function to show a region selector popup.
    
    Args:
        image: The input image as a numpy array
        params: Dictionary with 'x', 'y', 'width', 'height' keys
        on_confirm: Callback function called with new params when OK is clicked
        on_cancel: Optional callback function called when Cancel is clicked
        max_display_size: Maximum display size for the image
    """
    selector = RegionSelector()
    selector.show(image, params, on_confirm, on_cancel, max_display_size)
