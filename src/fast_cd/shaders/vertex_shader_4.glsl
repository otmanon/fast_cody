#version 150
 uniform mat4 view;
 uniform mat4 proj;
 uniform mat4 normal_matrix;
 uniform mat4x3 primary_bones[4];
 uniform mat4x3 secondary_bones[4];

 in vec3 position;
 in vec3 normal;
 out vec3 position_eye;
 out vec3 normal_eye;
 in vec4 Ka;
 in vec4 Kd;
 in vec4 Ks;
 in vec4 primary_weights_1;

 in vec4 secondary_weights_1;
 //in mat4 secondary_weights;
 in vec2 texcoord;
 out vec2 texcoordi;
 out vec4 Kai;
 out vec4 Kdi;
 out vec4 Ksi;
 out float w;
 void main()
 {
     vec4 x = vec4(position, 1.0);
     vec4 ones = vec4(0.0, 0.0,0.0, 1.0);
     mat4x3 P = mat4x3(0.0);
     for (int i = 0; i < 4; i++)
     {
         P += primary_weights_1[i] * primary_bones[i];
         P += secondary_weights_1[i] * secondary_bones[i];
     }
     vec3 p =  P * x;

     vec3 n = P* vec4(normal, 0.0);
     //vec3 p = primary_bones[0][3] + position;
     position_eye =   vec3 (view  * vec4(p, 1.0));
    // position_eye = vec3 (view * vec4 (position, 1.0));
     normal_eye =vec3(normal_matrix * vec4 (n, 0.0));
     normal_eye = normalize(normal_eye);
     gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
     Kai = Ka;
     Kdi = Kd;
     Ksi = Ks;
     texcoordi = texcoord;
     w = 1;
}
