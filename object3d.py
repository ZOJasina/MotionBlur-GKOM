from OpenGL.GL import *
import glm

class Object3D:
    def __init__(self, vao, vbo, draw_commands, texture_ids=[], material_properties=None):
        """
        Initialize a 3D object with required OpenGL identifiers.

        Parameters:
        - vao: int, Vertex Array Object ID
        - vbo: int, Vertex Buffer Object ID
        - draw_commands: list, instructions for rendering
        - texture_ids: list of texture IDs for each material
        - material_properties: list of dicts with material properties for each material
        """
        self.vao = vao
        self.vbo = vbo
        self.draw_commands = draw_commands
        self.texture_ids = texture_ids
        self.material_properties = material_properties if material_properties else []
        self.deleted = False  # to avoid double deletion

        # Material properties (default values) - used when material_properties is not set
        self.diffuse = [1.0, 1.0, 1.0]    # RGB  #Don't override material.diffuse if we want to use textures
        self.specular = [1.0, 1.0, 1.0]   # RGB
        self.shininess = 32.0
        self.ambient = [0.1, 0.1, 0.1]    # RGB
        self.opacity = 1.0

        # Transformation properties (default values)
        self.translation = [0.0, 0.0, 0.0]
        self.scale_factor = 1.0

    # Fluent setters
    def translate(self, x, y, z):
        """Translate the object by a given vector."""
        self.translation[0] += x
        self.translation[1] += y
        self.translation[2] += z
        return self

    def scale(self, factor):
        """Scale the object by a given factor."""
        self.scale_factor *= factor
        return self

    def set_material(self, diffuse=None, specular=None, shininess=None, ambient=None, opacity=None):
        """Update the material properties of the object."""
        if diffuse:
            self.diffuse = diffuse
        if specular:
            self.specular = specular
        if ambient:
            self.ambient = ambient
        if shininess is not None:
            self.shininess = shininess
        if opacity is not None:
            self.opacity = opacity
        return self

    def prepare_material(self, shader_program):
        glUniform3f(glGetUniformLocation(shader_program, "material.diffuse"), self.diffuse[0], self.diffuse[1], self.diffuse[2])
        glUniform3f(glGetUniformLocation(shader_program, "material.specular"), self.specular[0], self.specular[1], self.specular[2])
        glUniform1f(glGetUniformLocation(shader_program, "material.shininess"), self.shininess)
        glUniform3f(glGetUniformLocation(shader_program, "material.ambient"), self.ambient[0], self.ambient[1], self.ambient[2])
        glUniform1f(glGetUniformLocation(shader_program, "material.opacity"), self.opacity)
        return

    def get_trans_matrix(self):
        tmatrix = glm.mat4(1.0)
        tmatrix = glm.translate(tmatrix, glm.vec3(self.translation[0], self.translation[1], self.translation[2]))
        # tmatrix = glm.rotate(tmatrix, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        tmatrix = glm.scale(tmatrix, glm.vec3(self.scale_factor))
        return tmatrix

    def get_position(self):
        return glm.vec3(self.translation[0], self.translation[1], self.translation[2])

    def draw(self):
        """
        Placeholder for draw logic.
        In a real OpenGL application, you would bind VAO,
        set uniforms for material and transformations,
        and issue the draw calls here.
        """
        print(f"Drawing Object3D with VAO {self.vao} and VBO {self.vbo}")
        print(f"Material: diffuse={self.diffuse}, specular={self.specular}, "
              f"ambient={self.ambient}, shininess={self.shininess}")
        print(f"Transform: translation={self.translation}, scale={self.scale_factor}")
        # Actual OpenGL draw calls would go here
        return self

    def delete(self):
        """Manually delete GPU resources for this object."""
        if not self.deleted:
            print(f"Deleting VAO {self.vao} and VBO {self.vbo}")
            glDeleteVertexArrays(1, [self.vao])
            glDeleteBuffers(1, [self.vbo])
            self.deleted = True

    def __del__(self):
        """Destructor: automatically called when the object is garbage collected."""
        self.delete()


