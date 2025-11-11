import glfw
from OpenGL.GL import *
import glm
import ctypes
from vertex_models import *
from car import Car
from PIL import Image
import numpy as np
import os


def load_shader(shader_file, shader_type):
    """reads and compiles shader from file"""
    try:
        with open(shader_file, "r") as f:
            shader_source = f.read()
    except FileNotFoundError:
        print(f"Error: file not found {shader_file}")
        return None

    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode("utf-8")
        print(f"Error compiling shader ({shader_file}):\n{log}")
        return None

    return shader


def create_shader_program(vertex_file, fragment_file):
    """Creates shader program"""
    vertex_shader = load_shader(vertex_file, GL_VERTEX_SHADER)
    fragment_shader = load_shader(fragment_file, GL_FRAGMENT_SHADER)

    if not vertex_shader or not fragment_shader:
        return None

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode("utf-8")
        print(f"Error linking program:\n{log}")
        return None

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program


def create_default_texture():
    """Creates a default white 1x1 texture."""
    texture_id = glGenTextures(1)
    if texture_id == 0:
        print("Failed to generate default texture ID")
        return None

    glBindTexture(GL_TEXTURE_2D, texture_id)

    white_pixel = np.array([255, 255, 255, 255], dtype=np.uint8)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, white_pixel)

    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id


def load_texture(texture_path):
    """Loads a texture from file and returns OpenGL texture ID."""
    if not texture_path or not os.path.exists(texture_path):
        print(f"Texture path does not exist: {texture_path}")
        return None

    try:
        img = Image.open(texture_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        img_data = np.array(img, dtype=np.uint8)
        width, height = img.size

        img_data = np.flipud(img_data)

        texture_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        error = glGetError()
        if error != GL_NO_ERROR:
            print(
                f"OpenGL error after loading texture {texture_path}: {error}")
            glDeleteTextures(1, [texture_id])
            glBindTexture(GL_TEXTURE_2D, 0)
            return None

        glBindTexture(GL_TEXTURE_2D, 0)
        print(
            f"Successfully loaded texture: {texture_path} (ID: {texture_id})")
        return texture_id

    except Exception as e:
        print(f"Error loading texture {texture_path}: {e}")
        return None


def setup_model(vertices_np):
    """Creates VAO + VBO for given vertex array.
    Expected format: [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, tex_u, tex_v] (8 floats per vertex)"""
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL_STATIC_DRAW)

    stride = 8 * vertices_np.itemsize
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao, vbo


def render_model(shader, vao, vertex_count, model, view, projection):
    """Renders a model with given transformation matrices."""
    glUseProgram(shader)
    normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))

    # Set uniforms
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix3fv(glGetUniformLocation(shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normalMatrix))

    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)
    glBindVertexArray(0)
    return


def render_complex_model(shader, vao, draw_commands, model, view, projection, texture_ids=None, default_texture=None):
    """Renders a complex (batched) model using draw commands"""
    glUseProgram(shader)

    # Set uniforms
    normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix3fv(glGetUniformLocation(shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normalMatrix))

    glBindVertexArray(vao)

    texture_loc = glGetUniformLocation(shader, "texture_diffuse1")
    use_texture_loc = glGetUniformLocation(shader, "useTexture")

    if texture_loc != -1:
        glUniform1i(texture_loc, 0)

    for idx, (start_index, vertex_count) in enumerate(draw_commands):
        glActiveTexture(GL_TEXTURE0)

        tex_id = None
        if texture_ids and idx < len(texture_ids) and texture_ids[idx] is not None:
            try:
                tex_id = int(texture_ids[idx])
            except (ValueError, TypeError):
                tex_id = None

        if tex_id is not None and tex_id > 0:
            glBindTexture(GL_TEXTURE_2D, tex_id)
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 1)
        else:
            if default_texture is not None and default_texture > 0:
                glBindTexture(GL_TEXTURE_2D, default_texture)
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 0)

        glDrawArrays(GL_TRIANGLES, start_index, vertex_count)

    glBindTexture(GL_TEXTURE_2D, 0)
    glBindVertexArray(0)
    return


def load_model(path):
    """Loads a model and returns (vao, vbo, draw_commands, texture_ids)"""
    vertices_np, draw_commands, texture_paths = load_model_batched(path)
    if vertices_np is None:
        print("Failed to load model, terminating.")
        glfw.terminate()
        return None, None, None, None

    vao, vbo = setup_model(vertices_np)

    # Load textures
    texture_ids = []
    for idx, tex_path in enumerate(texture_paths):
        if tex_path:
            tex_id = load_texture(tex_path)
            texture_ids.append(tex_id)
            if tex_id is None:
                print(
                    f"Warning: Failed to load texture {idx} for model {path}: {tex_path}")
            else:
                print(
                    f"Loaded texture {idx} for model {path}: ID={tex_id}, path={tex_path}")
        else:
            texture_ids.append(None)
            print(f"Info: No texture path for material {idx} in model {path}")

    loaded_count = sum(1 for tid in texture_ids if tid is not None and tid > 0)
    print(
        f"Loaded {loaded_count}/{len(texture_paths)} textures for model {path}")
    print(f"Texture IDs for {path}: {texture_ids}")
    return vao, vbo, draw_commands, texture_ids


def main():
    if not glfw.init():
        print("Cannot initiate GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Motion Blur", None, None)
    if not window:
        glfw.terminate()
        print("Cannot create a window")
        return

    glfw.make_context_current(window)

    shader_program = create_shader_program("phong.vert", "phong.frag")
    glEnable(GL_DEPTH_TEST)
    if not shader_program:
        glfw.terminate()
        return

    default_texture = create_default_texture()

    # Load batched model
    car_vao, car_vbo, car_draw_commands, car_texture_ids = load_model(
        "objects/track-road-narrow-straight-bend.obj")
    road_vao, road_vbo, road_draw_commands, road_texture_ids = load_model(
        "objects/track-road-narrow-straight-bump-up.obj")
    pine_tree_left_vao, pine_tree_left_vbo, pine_tree_left_draw_commands, pine_tree_left_texture_ids = load_model(
        "objects/pine_tree.obj")
    green_tree_right_vao, green_tree_right_vbo, green_tree_right_draw_commands, green_tree_right_texture_ids = load_model(
        "objects/green_tree.obj")


    glClearColor(0.1, 0.1, 0.1, 1.0)

    # Set up light and camera
    glUseProgram(shader_program)
    view = glm.lookAt(glm.vec3(0, 0, 2), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    projection = glm.perspective(glm.radians(45.0), fb_w / fb_h if fb_h != 0 else 4/3, 0.1, 100.0)

    def uni(name):
        loc = glGetUniformLocation(shader_program, name)
        if loc == -1:
            print(f"Warning: uniform '{name}' not found (location = -1). Maybe it's optimized out or name mismatch.")
        return loc

    glUniform3f(uni("lightPos"), 1.2, 1.0, 2.0)
    glUniform3f(uni("viewPos"), 0.0, 0.0, 2.0)

    # Material
    glUniform3f(uni("material.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(uni("material.diffuse"), 1.0, 0.5, 0.31)
    glUniform3f(uni("material.specular"), 0.5, 0.5, 0.5)
    glUniform1f(uni("material.shininess"), 32.0)

    # Light
    glUniform3f(uni("light.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(uni("light.diffuse"), 0.8, 0.8, 0.8)
    glUniform3f(uni("light.specular"), 1.0, 1.0, 1.0)

    # Car
    car = Car(initial_position=glm.vec3(-0.1, -0.35, 0.0))
    previous_time = glfw.get_time() # Czas ostatniej klatki

    # Camera Mode
    camera_mode = 0  # 0: 3rd POV, 1: 1st POV
    def key_callback(window, key, scancode, action, mods):
        nonlocal camera_mode, height_offset
        if key == glfw.KEY_C and action == glfw.PRESS:
            camera_mode = 1 - camera_mode
    glfw.set_key_callback(window, key_callback)

    # Main loop
    while not glfw.window_should_close(window):
        # handle window resize
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fb_w, fb_h)
        # Update projection matrix on resize
        projection = glm.perspective(glm.radians(45.0), fb_w / fb_h if fb_h != 0 else 4/3, 0.1, 100.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        time_val = glfw.get_time()
        delta_time = time_val - previous_time
        previous_time = time_val
        car.update(window, delta_time)

        # === CAMERA ===
        up = glm.vec3(0.0, 1.0, 0.0)
        forward_x, forward_z = -glm.sin(car.angle), -glm.cos(car.angle)
        forward_vector = glm.vec3(forward_x, 0.0, forward_z)

        if camera_mode == 0:
            backward_vector = -forward_vector
            distance_behind, height_above, look_ahead = 2.0, 0.5, 1.0
            eye = car.position + (backward_vector * distance_behind) + glm.vec3(0.0, height_above, 0.0)
            center = car.position + (forward_vector * look_ahead)
        else:
            right_vector = glm.vec3(-glm.cos(car.angle), 0.0, glm.sin(car.angle))
            height_offset, right_offset, forward_offset, look_ahead = 0.08, 0.06, -0.075, 1.0
            eye = car.position + glm.vec3(0.0, height_offset, 0.0) + (right_vector * right_offset) + (forward_vector * forward_offset)
            center = eye + (forward_vector * look_ahead)

        view = glm.lookAt(eye, center, up)

        # === CAR ===
        # Don't override material.diffuse if we want to use textures
        glUniform3f(uni("material.diffuse"), 1.0, 1.0, 1.0)
        glUniform3f(uni("material.specular"), 0.9, 0.9, 0.9)
        glUniform1f(uni("material.shininess"), 64.0)

        model_car = car.get_model_matrix()
        # model_car = glm.scale(model_car, glm.vec3(0.2))
        # model_car = glm.rotate(model_car, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        # model_car = glm.translate(model_car, glm.vec3(-0.1, -0.35, 0.0))
        render_complex_model(shader_program, car_vao, car_draw_commands,
                             model_car, view, projection, car_texture_ids, default_texture)

        # === ROAD ===
        glUniform3f(uni("material.diffuse"), 0.5, 0.5, 0.5)
        glUniform3f(uni("material.specular"), 0.1, 0.1, 0.1)
        glUniform1f(uni("material.shininess"), 8.0)

        model_road = glm.mat4(1.0)
        model_road = glm.translate(model_road, glm.vec3(0.0, -0.5, 0.0))
        # model_road = glm.rotate(model_road, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_road = glm.scale(model_road, glm.vec3(0.2))
        render_complex_model(shader_program, road_vao, road_draw_commands,
                             model_road, view, projection, road_texture_ids, default_texture)

         # === LEFT TREE (pine) ===
        glUniform3f(uni("material.diffuse"), 0.0, 0.4, 0.0)
        glUniform3f(uni("material.specular"), 0.2, 0.2, 0.2)
        glUniform1f(uni("material.shininess"), 16.0)

        model_pine_tree_left = glm.mat4(1.0)
        model_pine_tree_left = glm.translate(model_pine_tree_left, glm.vec3(-0.7, -0.5, -1.0))
        # model_pine_tree_left = glm.rotate(model_pine_tree_left, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_pine_tree_left = glm.scale(model_pine_tree_left, glm.vec3(0.15))
        render_complex_model(shader_program, pine_tree_left_vao, pine_tree_left_draw_commands,
                             model_pine_tree_left, view, projection, pine_tree_left_texture_ids, default_texture)

            # === RIGHT TREE (green) ===
        glUniform3f(uni("material.diffuse"), 0.2, 0.8, 0.2)
        glUniform3f(uni("material.specular"), 0.3, 0.3, 0.3)
        glUniform1f(uni("material.shininess"), 32.0)

        model_green_tree_right = glm.mat4(0.3)
        model_green_tree_right = glm.translate(model_green_tree_right, glm.vec3(0.5, -0.5, 0.0))
        # model_green_tree_right = glm.rotate(model_green_tree_right, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_green_tree_right = glm.scale(
            model_green_tree_right, glm.vec3(0.2))
        render_complex_model(shader_program, green_tree_right_vao, green_tree_right_draw_commands,
                             model_green_tree_right, view, projection, green_tree_right_texture_ids, default_texture)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glDeleteVertexArrays(1, [car_vao])
    glDeleteBuffers(1, [car_vbo])
    glDeleteVertexArrays(1, [road_vao])
    glDeleteBuffers(1, [road_vbo])
    glDeleteVertexArrays(1, [pine_tree_left_vao])
    glDeleteBuffers(1, [pine_tree_left_vbo])
    glDeleteVertexArrays(1, [green_tree_right_vao])
    glDeleteBuffers(1, [green_tree_right_vbo])

    glDeleteProgram(shader_program)

    glfw.terminate()


if __name__ == "__main__":
    main()
