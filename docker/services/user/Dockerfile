FROM runtime AS runtime-final

ARG UID=1000
ARG GID=1000
ARG USERNAME=colmap

RUN groupadd --gid ${GID} ${USERNAME} \
 && useradd  --uid ${UID} --gid ${GID} -m -s /bin/bash ${USERNAME} \
 && mkdir -p /workspace \
 && chown -R ${UID}:${GID} /workspace /home/${USERNAME}

# Add sudo access to the user
RUN echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER ${USERNAME}