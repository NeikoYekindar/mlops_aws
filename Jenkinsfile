pipeline {
    agent any
    environment {
        // Thay thế bằng tên S3 bucket của bạn
        S3_BUCKET = 'my-weather-pipeline-artifacts'
        IMAGE_NAME = 'weather-tcn-trainer'
        // Tự động tạo tag image dựa trên số lần build của Jenkins
        IMAGE_TAG = "build-${BUILD_NUMBER}"
        // Đặt region cho AWS SDK/CLI bên trong container (điều chỉnh phù hợp)
        AWS_DEFAULT_REGION = 'ap-southeast-1'
    }

    stages {
        stage('Checkout Code'){
            steps{
                git branch: 'main', url: 'https://github.com/NeikoYekindar/mlops_aws.git'
            }
        }
        stage('Build Docker Image'){
            steps{
                script {
                    // Build image; assume docker is available on the agent
                    sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
                }
            }
        }
        stage('Train model 1'){
            steps{
                script{
                    sh 'mkdir -p model'
                    withCredentials([
                        string(credentialsId: 'AWS_ACCESS_KEY_ID', variable: 'AWS_ACCESS_KEY_ID'),
                        string(credentialsId: 'AWS_SECRET_ACCESS_KEY', variable: 'AWS_SECRET_ACCESS_KEY'),
                        
                    ]) {
                        sh """
                        docker run --rm \
                            -e AWS_ACCESS_KEY_ID=\"$AWS_ACCESS_KEY_ID\" \
                            -e AWS_SECRET_ACCESS_KEY=\"$AWS_SECRET_ACCESS_KEY\" \
                            -e AWS_SESSION_TOKEN=\"$AWS_SESSION_TOKEN\" \
                            -e AWS_DEFAULT_REGION=\"${AWS_DEFAULT_REGION}\" \
                            -v \$(pwd)/model:/app/model \
                            ${IMAGE_NAME}:${IMAGE_TAG} \
                            weather_train_1.py \
                            --data-path s3://${S3_BUCKET}/data/weather_dataset.csv \
                            --model-output-path /app/model/model_1.pth
                        """
                    }
                }
            }
        }
        stage('Train model 2'){
            steps{
                script{
                    withCredentials([
                        string(credentialsId: 'AWS_ACCESS_KEY_ID', variable: 'AWS_ACCESS_KEY_ID'),
                        string(credentialsId: 'AWS_SECRET_ACCESS_KEY', variable: 'AWS_SECRET_ACCESS_KEY'),
                        
                    ]) {
                        sh """
                        docker run --rm \
                            -e AWS_ACCESS_KEY_ID=\"$AWS_ACCESS_KEY_ID\" \
                            -e AWS_SECRET_ACCESS_KEY=\"$AWS_SECRET_ACCESS_KEY\" \
                            -e AWS_SESSION_TOKEN=\"$AWS_SESSION_TOKEN\" \
                            -e AWS_DEFAULT_REGION=\"${AWS_DEFAULT_REGION}\" \
                            -v \$(pwd)/model:/app/model \
                            ${IMAGE_NAME}:${IMAGE_TAG} \
                            weather_train_2.py \
                            --data-path s3://${S3_BUCKET}/data/weather_dataset.csv \
                            --model-output-path /app/model/model_2.pth
                        """
                    }
                }
            }
        }
        stage('Evaluate & Select Best'){
            steps{
                script{
                    withCredentials([
                        string(credentialsId: 'AWS_ACCESS_KEY_ID', variable: 'AWS_ACCESS_KEY_ID'),
                        string(credentialsId: 'AWS_SECRET_ACCESS_KEY', variable: 'AWS_SECRET_ACCESS_KEY'),
                        
                    ]) {
                        sh """
                        docker run --rm \
                            -e AWS_ACCESS_KEY_ID=\"$AWS_ACCESS_KEY_ID\" \
                            -e AWS_SECRET_ACCESS_KEY=\"$AWS_SECRET_ACCESS_KEY\" \
                            -e AWS_SESSION_TOKEN=\"$AWS_SESSION_TOKEN\" \
                            -e AWS_DEFAULT_REGION=\"${AWS_DEFAULT_REGION}\" \
                            -v \$(pwd)/model:/app/model \
                            ${IMAGE_NAME}:${IMAGE_TAG} \
                            weather_test.py \
                            --data-path s3://${S3_BUCKET}/data/weather_test.csv \
                            --model1-path /app/model/model_1.pth \
                            --model2-path /app/model/model_2.pth \
                            --out-dir /app/model
                        """
                    }
                }
            }
        }
        stage('Archive & Upload artifacts'){
            steps{
                script {
                    archiveArtifacts artifacts: 'model/**', fingerprint: true
                    withCredentials([
                        string(credentialsId: 'AWS_ACCESS_KEY_ID', variable: 'AWS_ACCESS_KEY_ID'),
                        string(credentialsId: 'AWS_SECRET_ACCESS_KEY', variable: 'AWS_SECRET_ACCESS_KEY'),
                        
                    ]) {
                        sh "aws s3 cp --region ${AWS_DEFAULT_REGION} model/best_model.pth s3://${S3_BUCKET}/model/best_model_${IMAGE_TAG}.pth"
                        sh "aws s3 cp --region ${AWS_DEFAULT_REGION} model/results.json s3://${S3_BUCKET}/model/results_${IMAGE_TAG}.json"
                    }
                }
            }
        }
        
    }
    post {
        always {
            echo 'Clearing up workspace'
            cleanWs()
            script {
                sh "docker rmi -f ${IMAGE_NAME}:${IMAGE_TAG} || true"
            }
        }
    }
}
