use glow::HasContext;

pub struct GpuJob {
    pub gl: std::sync::Arc<glow::Context>,
    pub prog: glow::NativeProgram,
    pub in_buffer: glow::NativeBuffer,
    pub out_buffer: glow::NativeBuffer,
}

impl GpuJob {
    pub unsafe fn new(
        gl: &std::sync::Arc<glow::Context>,
        src: &str,
        in_size: usize,
        out_size: usize,
    ) -> Self {
        let cs = gl.create_shader(glow::COMPUTE_SHADER).unwrap();
        gl.shader_source(cs, src);
        gl.compile_shader(cs);
        assert!(gl.get_shader_compile_status(cs), "compile log: {}", gl.get_shader_info_log(cs));

        let prog = gl.create_program().unwrap();
        gl.attach_shader(prog, cs);
        gl.link_program(prog);
        assert!(gl.get_program_link_status(prog));

        let in_buffer = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(in_buffer));
        gl.buffer_data_size(glow::SHADER_STORAGE_BUFFER, in_size as i32, glow::DYNAMIC_DRAW);

        let out_buffer = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(out_buffer));
        gl.buffer_data_size(glow::SHADER_STORAGE_BUFFER, out_size as i32, glow::DYNAMIC_DRAW);

        Self {
            gl: gl.clone(),
            prog,
            in_buffer,
            out_buffer,
        }
    }

    pub unsafe fn dispatch(&self, x: u32, y: u32, z: u32) {
        self.gl.use_program(Some(self.prog));
        self
            .gl
            .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(self.in_buffer));
        self
            .gl
            .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(self.out_buffer));
        self.gl.dispatch_compute(x, y, z);
        self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
    }

    pub unsafe fn read_ssbo_u8(&self, size: usize) -> Vec<u8> {
        self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.out_buffer));
        let mut data = vec![0u8; size];
        self.gl.get_buffer_sub_data(glow::SHADER_STORAGE_BUFFER, 0, &mut data);
        data
    }

    pub unsafe fn update_ssbo_data(&self, data: &[u8]) {
        self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.in_buffer));
        self.gl.buffer_sub_data_u8_slice(glow::SHADER_STORAGE_BUFFER, 0, data);
    }
}
