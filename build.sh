#!/bin/zsh
# Build SiliconFly
cd "$(dirname "$0")"
swiftc -O -swift-version 5 -o SiliconFly main.swift FlyModel.swift Sim.swift MetalSim.swift GPUCheck.swift Diagnostics.swift \
    BrainView.swift Environment.swift LabServer.swift -framework Cocoa -framework SceneKit -framework Metal || exit 1
echo "Built ./SiliconFly"
