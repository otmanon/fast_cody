#version 150
 uniform mat4 view;
 uniform mat4 proj;
 uniform vec4 fixed_color;
 in vec3 position_eye;
 in vec3 normal_eye;
 uniform vec3 light_position_eye;
 vec3 Ls = vec3 (1, 1, 1);
 vec3 Ld = vec3 (1, 1, 1);
 vec3 La = vec3 (1, 1, 1);
 in vec4 Ksi;
 in vec4 Kdi;
 in vec4 Kai;
 in float w;
 in vec2 texcoordi;
 uniform sampler2D tex;
 uniform float specular_exponent;
 uniform float lighting_factor;
 uniform float texture_factor;
 uniform float matcap_factor;
 uniform float double_sided;
 out vec4 outColor;
 void main()
 {
//     vec3 xTangent = dFdx( position_eye );
//     vec3 yTangent = dFdy( position_eye );
//     vec3 normal_eye = normalize( cross( xTangent, yTangent ) );
     if(matcap_factor == 1.0f)
     {
         vec2 uv = normalize(normal_eye).xy * 0.5 + 0.5;
         outColor = texture(tex, uv);
     }else
     {
         vec3 Ia = La * vec3(Kai);    // ambient intensity

         vec3 vector_to_light_eye = light_position_eye - position_eye;
         vec3 direction_to_light_eye = normalize (vector_to_light_eye);
         float dot_prod = dot (direction_to_light_eye, normalize(normal_eye));
         float clamped_dot_prod = abs(max (dot_prod, -double_sided));
         vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity

         vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
         vec3 surface_to_viewer_eye = normalize (-position_eye);
         float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
         dot_prod_specular = float(abs(dot_prod)==dot_prod) * abs(max (dot_prod_specular, -double_sided));
         float specular_factor = pow (dot_prod_specular, specular_exponent);
         vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
         vec4 color = vec4(lighting_factor * (Is + Id) + Ia + (1.0-lighting_factor) * vec3(Kdi),(Kai.a+Ksi.a+Kdi.a)/3);
         outColor = mix(vec4(1,1,1,1), texture(tex, texcoordi), texture_factor) * color;
         if (fixed_color != vec4(0.0)) outColor = fixed_color;
         outColor.x *= w;
         outColor.y *= w;
         outColor.z *= w;
     }
 }
