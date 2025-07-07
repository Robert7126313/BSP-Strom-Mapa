use glow::HasContext;

pub struct GpuJob {
    pub gl: std::sync::Arc<glow::Context>,
    pub prog: glow::NativeProgram,
    pub ssbo: glow::NativeBuffer,
}

impl GpuJob {
    pub unsafe fn new(gl: &std::sync::Arc<glow::Context>, src: &str, ssbo_size: usize) -> Self {
        let cs = gl.create_shader(glow::COMPUTE_SHADER).unwrap();
        gl.shader_source(cs, src);
        gl.compile_shader(cs);
        assert!(gl.get_shader_compile_status(cs), "compile log: {}", gl.get_shader_info_log(cs));

        let prog = gl.create_program().unwrap();
        gl.attach_shader(prog, cs);
        gl.link_program(prog);
        assert!(gl.get_program_link_status(prog));

        let ssbo = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(ssbo));
        gl.buffer_data_size(glow::SHADER_STORAGE_BUFFER, ssbo_size as i32, glow::DYNAMIC_DRAW);

        Self { gl: gl.clone(), prog, ssbo }
    }

    pub unsafe fn dispatch(&self, x: u32, y: u32, z: u32) {
        self.gl.use_program(Some(self.prog));
        self.gl.dispatch_compute(x, y, z);
        self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
    }

    pub unsafe fn read_ssbo_u8(&self, size: usize) -> Vec<u8> {
        self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.ssbo));
        let mut data = vec![0u8; size];
        self.gl.get_buffer_sub_data(glow::SHADER_STORAGE_BUFFER, 0, &mut data);
        data
    }

    pub unsafe fn update_ssbo_data(&self, data: &[u8]) {
        self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.ssbo));
        self.gl.buffer_sub_data_u8_slice(glow::SHADER_STORAGE_BUFFER, 0, data);
    }
}
