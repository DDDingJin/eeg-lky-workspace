from __future__ import annotations

import torch
from torch import nn


class EEGNetRegressor(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 63,
        input_length: int = 50,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_input_channels = num_input_channels

        self.firstconv = nn.Conv2d(1, temporal_filters, (1, 3), padding=(0, 1), bias=False)
        self.depthwise = nn.Conv2d(
            temporal_filters,
            temporal_filters * depth_multiplier,
            (num_input_channels, 1),
            groups=temporal_filters,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(temporal_filters * depth_multiplier)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 2))
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.separable_depth = nn.Conv2d(
            temporal_filters * depth_multiplier,
            temporal_filters * depth_multiplier,
            (1, 3),
            padding=(0, 1),
            groups=temporal_filters * depth_multiplier,
            bias=False,
        )
        self.separable_point = nn.Conv2d(
            temporal_filters * depth_multiplier,
            separable_filters,
            (1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(separable_filters)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 2))
        self.drop2 = nn.Dropout2d(dropout_rate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_input_channels, input_length)
            feat = self._forward_features(dummy)
        self.head = nn.Linear(feat.numel(), 1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        return self.head(x).flatten()


class VLAAILiteRegressor(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 63,
        input_length: int = 50,
        extractor_filters: tuple[int, ...] = (64, 64, 64, 32, 32),
        extractor_kernel: int = 3,
        context_kernel: int = 8,
        nb_blocks: int = 3,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_input_channels = num_input_channels
        self.use_skip = use_skip

        self.extractors = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.contexts = nn.ModuleList()

        for block_idx in range(nb_blocks):
            in_channels = num_input_channels if block_idx == 0 else num_input_channels
            layers: list[nn.Module] = []
            current_channels = in_channels
            for out_channels in extractor_filters:
                layers.extend(
                    [
                        nn.Conv1d(current_channels, out_channels, extractor_kernel, padding=extractor_kernel // 2),
                        nn.BatchNorm1d(out_channels),
                        nn.LeakyReLU(),
                    ]
                )
                current_channels = out_channels
            self.extractors.append(nn.Sequential(*layers))
            self.projections.append(nn.Conv1d(current_channels, num_input_channels, kernel_size=1))
            self.contexts.append(
                nn.Sequential(
                    nn.ConstantPad1d((context_kernel - 1, 0), 0.0),
                    nn.Conv1d(num_input_channels, num_input_channels, kernel_size=context_kernel),
                    nn.LeakyReLU(),
                )
            )

        self.head = nn.Linear(num_input_channels * input_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.zeros_like(x) if self.use_skip else x
        for extractor, projection, context in zip(self.extractors, self.projections, self.contexts):
            source = x + residual if self.use_skip else residual
            out = extractor(source)
            out = projection(out)
            residual = context(out)
        return self.head(residual.flatten(start_dim=1)).flatten()


class _VLAAIExtractorExact(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_length: int,
        filters: tuple[int, ...],
        kernels: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        current_length = input_length
        for out_channels, kernel in zip(filters, kernels):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=0))
            current_length = current_length - kernel + 1
            layers.append(nn.LayerNorm([out_channels, current_length]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.ConstantPad1d((0, kernel - 1), 0.0))
            current_length = current_length + kernel - 1
            in_channels = out_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _VLAAIContextExact(nn.Module):
    def __init__(self, input_channels: int, filter_: int, kernel: int, input_length: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.ConstantPad1d((kernel - 1, 0), 0.0),
            nn.Conv1d(input_channels, filter_, kernel_size=kernel, padding=0),
            nn.LayerNorm([filter_, input_length]),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VLAAIExactOfficialRegressor(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 63,
        input_length: int = 50,
        nb_blocks: int = 4,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_input_channels = num_input_channels
        self.internal_input_channels = 64
        self.use_skip = use_skip

        extractor_filters = (256, 256, 256, 128, 128)
        extractor_kernels = (8, 8, 8, 8, 8)
        context_filter = 64
        context_kernel = 32

        self.extractors = nn.ModuleList(
            [
                _VLAAIExtractorExact(
                    input_channels=self.internal_input_channels,
                    input_length=input_length,
                    filters=extractor_filters,
                    kernels=extractor_kernels,
                )
                for _ in range(nb_blocks)
            ]
        )
        self.projections = nn.ModuleList([nn.Conv1d(extractor_filters[-1], self.internal_input_channels, kernel_size=1) for _ in range(nb_blocks)])
        self.contexts = nn.ModuleList(
            [
                _VLAAIContextExact(
                    input_channels=self.internal_input_channels,
                    filter_=context_filter,
                    kernel=context_kernel,
                    input_length=input_length,
                )
                for _ in range(nb_blocks)
            ]
        )
        self.head = nn.Linear(context_filter * input_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 63:
            x = torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)], dim=1)
        state = torch.zeros_like(x) if self.use_skip else x
        for extractor, projection, context in zip(self.extractors, self.projections, self.contexts):
            current = extractor(x + state if self.use_skip else state)
            current = projection(current)
            state = context(current)
        return self.head(state.flatten(start_dim=1)).flatten()


class VLAAIExactADTRegressor(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 63,
        input_length: int = 50,
        nb_blocks: int = 4,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_input_channels = num_input_channels
        self.internal_input_channels = 64
        self.use_skip = use_skip

        extractor_filters = (256, 256, 256, 128, 128)
        extractor_kernels = (64, 64, 64, 64, 64)
        context_filter = 64
        context_kernel = 64

        self.extractors = nn.ModuleList(
            [
                _VLAAIExtractorExact(
                    input_channels=self.internal_input_channels,
                    input_length=input_length,
                    filters=extractor_filters,
                    kernels=extractor_kernels,
                )
                for _ in range(nb_blocks)
            ]
        )
        self.projections = nn.ModuleList([nn.Conv1d(extractor_filters[-1], self.internal_input_channels, kernel_size=1) for _ in range(nb_blocks)])
        self.contexts = nn.ModuleList(
            [
                _VLAAIContextExact(
                    input_channels=self.internal_input_channels,
                    filter_=context_filter,
                    kernel=context_kernel,
                    input_length=input_length,
                )
                for _ in range(nb_blocks)
            ]
        )
        self.head = nn.Linear(context_filter * input_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 63:
            x = torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)], dim=1)
        state = torch.zeros_like(x) if self.use_skip else x
        for extractor, projection, context in zip(self.extractors, self.projections, self.contexts):
            current = extractor(x + state if self.use_skip else state)
            current = projection(current)
            state = context(current)
        return self.head(state.flatten(start_dim=1)).flatten()


class ADTLiteRegressor(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 63,
        input_length: int = 50,
        temporal_filters: int = 8,
        temporal_kernel: int = 7,
        depth_multiplier: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_input_channels = num_input_channels
        embed_dim = temporal_filters * depth_multiplier

        self.temporal = nn.Conv2d(1, temporal_filters, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False)
        self.norm1 = nn.BatchNorm2d(temporal_filters)
        self.spatial = nn.Conv2d(
            temporal_filters,
            embed_dim,
            (num_input_channels, 1),
            groups=temporal_filters,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.act = nn.ELU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_length, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(embed_dim * input_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.norm1(x)
        x = self.spatial(x)
        x = self.norm2(x)
        x = self.act(x)
        x = x.squeeze(2).transpose(1, 2)
        x = x + self.pos_embedding[:, : x.shape[1]]
        x = self.dropout(self.transformer(x))
        return self.head(x.flatten(start_dim=1)).flatten()
