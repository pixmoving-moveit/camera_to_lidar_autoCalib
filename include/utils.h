#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

class Utils {
private:
    std::string current_path;
    YAML::Node config_node;
    std::string sensors_params_path;
    YAML::Node extrinsic_params;
    YAML::Node robobus_params;

    std::map<std::string, Vector6d> relativePositionMap; // camera_key -> relative_position
    std::map<std::string, std::string> camera_key_map;          // camera_key -> extrinsic_params_key
    std::vector<std::string> gt_sensors_to_front;
    std::vector<std::string> gt_sensors_to_rear;

public:
    // --- 参数初始化 ---
    bool init_params(const std::string& config_path) 
    {
        current_path = config_path;
        if (!readConfig(current_path)) return false;
        if (!readSensorsParams()) return false;
        if (!readRobobusGroundTruth()) return false;
        return true;
    }

    // --- 读取配置文件 ---
    bool readConfig(const std::string& path) {
        try {
            std::string config_path = path + "/sensor_kit_calibration_gt.yaml";
            std::cout << "Loading config from: " << config_path << std::endl;

            config_node = YAML::LoadFile(config_path);
            sensors_params_path = config_node["sensors_params"]["sensors_params_path"].as<std::string>();
            gt_sensors_to_front = config_node["gt_sensors_to_front"].as<std::vector<std::string>>();
            gt_sensors_to_rear = config_node["gt_sensors_to_rear"].as<std::vector<std::string>>();

            // 映射相机 Key
            for (auto& it : gt_sensors_to_front) camera_key_map[it] = config_node["key_map"][it].as<std::string>();
            for (auto& it : gt_sensors_to_rear)  camera_key_map[it] = config_node["key_map"][it].as<std::string>();

            std::cout << "Sensors params path: " << sensors_params_path << std::endl;
        } catch (const YAML::Exception& e) {
            std::cerr << "YAML Config Error: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    // --- 读取传感器外参 ---
    bool readSensorsParams() {
        std::string extr_params_path = sensors_params_path + "/extrinsic_parameters/sensor_kit_calibration.yaml";
        try {
            extrinsic_params = YAML::LoadFile(extr_params_path);
            std::cout << "Loaded extrinsic params: " << extr_params_path << std::endl;
        } catch (const YAML::Exception& e) {
            std::cerr << "YAML Extrinsic Error: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    bool getNodePart(const std::string& node_key, YAML::Node& node)
    {
        if(!config_node[node_key])
        {
            std::cerr << "YAML Node Error: " << node_key << " not found." << std::endl;
            return false;
        }
        node = config_node[node_key];
        return true;
    }

    // --- 读取真值数据 ---
    bool readRobobusGroundTruth() {
        try {
            robobus_params = config_node["origin"];
            // 测试计算
            Vector6d relative = getRelativePositionGT(robobus_params, "lidar_front_top", "camera_front");
            std::cout << "Sample relative GT: " << relative.transpose() << std::endl;
        } catch (const YAML::Exception& e) {
            std::cerr << "YAML GT Error: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    // --- 计算外参真值 ---
    bool getExtrinsicParamsGT() {
        for (auto& sensor : gt_sensors_to_front) {
            relativePositionMap[sensor] = getRelativePositionGT(robobus_params, sensor, "lidar_front_top") + 
                                          getSensorRelativePosition(extrinsic_params, "lidar_ft_base_link");
        }
        for (auto& sensor : gt_sensors_to_rear) {
            relativePositionMap[sensor] = getRelativePositionGT(robobus_params, sensor, "lidar_rear_top") + 
                                          getSensorRelativePosition(extrinsic_params, "lidar_rt_base_link");
        }

        std::cout << "--- Relative Position Map ---" << std::endl;
        for (const auto& [name, pos] : relativePositionMap) {
            std::cout << name << ": " << pos.transpose() << std::endl;
        }
        return true;
    }

    // --- 核心计算逻辑：获取相对位置 ---
    Vector6d getRelativePositionGT(const YAML::Node& origin_node, const std::string& from_sensor, const std::string& to_sensor) {
        if (!origin_node[from_sensor] || !origin_node[to_sensor]) {
            throw std::runtime_error("Sensor key missing in origin_node: " + from_sensor + " or " + to_sensor);
        }

        const auto& from = origin_node[from_sensor];
        const auto& to = origin_node[to_sensor];

        Vector6d relative;
        relative << from["x"].as<double>() - to["x"].as<double>(),
                    from["y"].as<double>() - to["y"].as<double>(),
                    from["z"].as<double>() - to["z"].as<double>(),
                    from["roll"].as<double>() - to["roll"].as<double>(),
                    from["pitch"].as<double>() - to["pitch"].as<double>(),
                    from["yaw"].as<double>() - to["yaw"].as<double>();
        return relative;
    }

    // --- 获取 Sensor Kit 内部偏移 ---
    Vector6d getSensorRelativePosition(const YAML::Node& origin_node, const std::string& to_sensor) 
    {
        if (!origin_node["sensor_kit_base_link"][to_sensor]) {
            throw std::runtime_error("Missing in sensor_kit_base_link: " + to_sensor);
        }
        const auto& node = origin_node["sensor_kit_base_link"][to_sensor];
        return Vector6d(node["x"].as<double>(), node["y"].as<double>(), node["z"].as<double>(), 0.0, 0.0, 0.0);
    }

    // --- 保存结果到 YAML ---
    bool saveExtrinsicParamsGT() {
        YAML::Node sensor_kit;
        for (auto& [sensor_name, pos] : relativePositionMap) {
            std::string key = camera_key_map[sensor_name];
            
            sensor_kit[key]["x"] = pos[0];
            sensor_kit[key]["y"] = pos[1];
            sensor_kit[key]["z"] = pos[2];
            
            // 保留原始的姿态角 (Roll, Pitch, Yaw)
            sensor_kit[key]["roll"]  = pos[3];
            sensor_kit[key]["pitch"] = pos[4];
            sensor_kit[key]["yaw"]   = pos[5];
        }

        config_node["sensor_kit_base_link"] = sensor_kit;
        std::string output_path = current_path + "/sensor_kit_calibration_gt.yaml";
        std::ofstream file(output_path);
        if (file.is_open()) {
            file << config_node;
            file.close();
            std::cout << "GT params saved to: " << output_path << std::endl;
            return true;
        }
        return false;
    }
};

#endif