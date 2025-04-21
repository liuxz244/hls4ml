from datetime import datetime

import numpy as np
from pynq import Overlay, allocate


class NeuralNetworkOverlay(Overlay):
    def __init__(
        self, bitfile_name, x_shape, y_shape, dtype=np.float32, dtbo=None, download=True, ignore_version=False, device=None
    ):
        super().__init__(bitfile_name, dtbo=None, download=True, ignore_version=False, device=None)
        self.sendchannel = self.hier_0.axi_dma_0.sendchannel
        self.recvchannel = self.hier_0.axi_dma_0.recvchannel
        self.input_buffer = allocate(shape=x_shape, dtype=dtype)
        self.output_buffer = allocate(shape=y_shape, dtype=dtype)

    def _print_dt(self, timea, timeb, N):
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 10**-6
        rate = N / dts
        print(f"Classified {N} samples in {dts} seconds ({rate} inferences / s)")
        return dts, rate

    def predict(self, X, debug=False, profile=False, encode=None, decode=None):
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)
        self.input_buffer[:] = X

        # 记录 DMA 开始时间戳
        dma_start = datetime.now()
        self.sendchannel.transfer(self.input_buffer)
        self.recvchannel.transfer(self.output_buffer)

        # 增加 DMA 传输监测点
        self.sendchannel.wait()
        dma_send_end = datetime.now()

        self.recvchannel.wait()
        dma_recv_end = datetime.now()

        if profile and debug:
            # 计算完整 DMA 传输时间
            full_dma_time = (dma_recv_end - dma_start).total_seconds()
            send_dma_time = (dma_send_end - dma_start).total_seconds()
            hw_compute_time = (dma_recv_end - dma_send_end).total_seconds()

            print(f"DMA 传输总耗时: {full_dma_time * 1000:.2f}ms")
            print(f"→ 输入传输耗时: {send_dma_time * 1000:.2f}ms")
            print(f"→ 硬件计算耗时: {hw_compute_time * 1000:.2f}ms")

        if decode is not None:
            self.output_buffer = decode(self.output_buffer)

        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            return self.output_buffer, dts, rate
        else:
            return self.output_buffer
