import glfw
from OpenGL.GL import *
import numpy as np


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

    vertices = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0]

    vertices_np = np.array(vertices, dtype=np.float32)

    shader_program = create_shader_program("simple.vert", "simple.frag")
    if not shader_program:
        glfw.terminate()
        return

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    glClearColor(0.0, 0.0, 0.0, 1.0)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)

        glBindVertexArray(vao)

        glDrawArrays(GL_TRIANGLES, 0, 3)

        glBindVertexArray(0)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteProgram(shader_program)

    glfw.terminate()


if __name__ == "__main__":
    main()
