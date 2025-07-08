pub const CULL_SHADER: &str = r#"
#version 430
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer InTris {
    vec4 vertices[];
};

layout(std430, binding = 1) writeonly buffer OutFlags {
    uint visible[];
};

uniform vec4 frustum[6];
uniform uint num_tris;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if(idx >= num_tris) {
        return;
    }
    uint base = idx * 3u;
    vec3 a = vertices[base].xyz;
    vec3 b = vertices[base + 1u].xyz;
    vec3 c = vertices[base + 2u].xyz;
    bool inside = true;
    for(int i = 0; i < 6; ++i) {
        vec4 p = frustum[i];
        if(dot(p.xyz, a) + p.w < 0.0 &&
           dot(p.xyz, b) + p.w < 0.0 &&
           dot(p.xyz, c) + p.w < 0.0) {
            inside = false;
            break;
        }
    }
    visible[idx] = inside ? 1u : 0u;
}
"#;
