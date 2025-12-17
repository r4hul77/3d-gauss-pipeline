FROM huggingface/transformers-pytorch-gpu

RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3

RUN python3 -m pip install --no-cache-dir -U pip \
 && python3 -m pip install --no-cache-dir -e /opt/sam3 \
 && python3 -m pip install --no-cache-dir -e "/opt/sam3[notebooks,train,dev]" \
 && python3 -m pip install --no-cache-dir decord

RUN python3 -c "import sam3; print('sam3 OK:', sam3.__file__)"