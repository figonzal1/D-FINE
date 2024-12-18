import time
import contextlib
import collections
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as T
import torchvision.ops as ops


import tensorrt as trt
import cv2  # Added for video processing
import os


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):

    def __init__(self, engine_path, device="cuda:0", backend="torch", verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend

        self.logger = trt.Logger(
            trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        try:
            self.engine = self.load_engine(engine_path)
        except Exception as e:
            raise RuntimeError(f"Error loading TensorRT engine: {str(e)}")

        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(
            self.engine, self.context, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr)
                                         for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_bindings(self, engine, context, device):
        Binding = collections.namedtuple(
            'Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = 16
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def run_torch(self, blob):
        for n in self.input_names:
            if blob[n].dtype is not self.bindings[n].data.dtype:
                blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(
                    shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, '{} dtype mismatch'.format(
                n)

        self.bindings_addr.update(
            {n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()


def draw(images, labels, boxes, scores, thrh=0.4, target_class=9):
    for i, im in enumerate(images):

        # Filtrar por umbral
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        # Filtrar por clase objetivo
        target_indices = (lab == target_class)
        lab = lab[target_indices]
        box = box[target_indices]
        scrs = scrs[target_indices]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline=(0, 255, 0))  # RGB Green

            # Calcular la posición y el tamaño del fondo del texto
            text = f"BOTELLA ({lab[j].item()}) - {round(scrs[j].item(), 2)}"

            # Usar textbbox para obtener el tamaño del texto
            bbox = draw.textbbox((b[0], b[1]), text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            padded_width = text_width + 2 * 5
            padded_height = text_height + 5

            # Dibujar el fondo verde detrás del texto
            text_box_position = (b[0], b[1] - padded_height)
            draw.rectangle(
                [text_box_position, (b[0] + padded_width, b[1])],
                fill=(0, 255, 0),  # Fondo verde
            )

            # Dibujar el texto sobre el fondo verde
            draw.text(
                text_box_position,
                text=text,
                fill='black',
                align=""
            )

    return images


def process_video(m, device):
    cap = cv2.VideoCapture("test-assets/video1.mp4")

    # Get video properties
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    fps_start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - fps_start_time)
            print(f"FPS: {fps:.2f}")

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)

        im_data = transforms(frame_pil)[None]

        blob = {
            'images': im_data.to(device),
            'orig_target_sizes': orig_size.to(device),
        }

        output = m(blob)

        # Extraer cajas, puntuaciones y etiquetas
        boxes = output['boxes']
        scores = output['scores']
        labels = output['labels']

        if boxes.numel() == 0:
            continue

        if scores.numel() == 0:
            continue

        # Draw detections on the frame
        result_images = draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(result_images[0]), cv2.COLOR_RGB2BGR)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    m = TRTInference("model.engine")

    process_video(m, "cuda:0")