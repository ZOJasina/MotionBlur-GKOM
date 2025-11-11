#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};
uniform Material material;

struct Light {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};
uniform Light light;

uniform sampler2D texture_diffuse1;
uniform int useTexture;

void main()
{
    vec3 baseColor = material.diffuse;
    if (useTexture == 1) {
        vec4 texColor = texture(texture_diffuse1, TexCoord);
        baseColor = texColor.rgb;
    }

    // Ambient
    vec3 ambient = light.ambient * material.ambient * baseColor;

    // Diffuse
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = light.diffuse * (diff * baseColor);

    // Specular (Phong)
    vec3 V = normalize(viewPos - FragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(R, V), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * material.specular);

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}