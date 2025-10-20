pipeline {
    agent any
    environment {
        S3_BUCKET = 'my-weather-pipeline-artifacts'
        IMAGE_NAME = 'weather-tcn-trainer'
        IMAGE_TAG = "build-${BUILD_NUMBER}"
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
        stage('Evaluate models 1'){
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
                            -e AWS_DEFAULT_REGION=\"${AWS_DEFAULT_REGION}\" \
                            -v \$(pwd)/model:/app/model \
                            ${IMAGE_NAME}:${IMAGE_TAG} \
                            weather_test_1.py \
                            --data-path s3://${S3_BUCKET}/data/weather_test_1.csv \
                            --model1-path /app/model/model_1.pth \
                            --model2-path /app/model/model_2.pth \
                            --out-dir /app/model
                        """
                    }
                }
            }
        }
        stage('Evaluate models 2'){
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
                            -e AWS_DEFAULT_REGION=\"${AWS_DEFAULT_REGION}\" \
                            -v \$(pwd)/model:/app/model \
                            ${IMAGE_NAME}:${IMAGE_TAG} \
                            weather_test_2.py \
                            --data-path s3://${S3_BUCKET}/data/weather_test_2.csv \
                            --model1-path /app/model/model_1.pth \
                            --model2-path /app/model/model_2.pth \
                            --out-dir /app/model
                        """
                    }
                }
            }
        }
        stage('Evaluate models 3'){
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
                            -e AWS_DEFAULT_REGION=\"${AWS_DEFAULT_REGION}\" \
                            -v \$(pwd)/model:/app/model \
                            ${IMAGE_NAME}:${IMAGE_TAG} \
                            weather_test_3.py \
                            --data-path s3://${S3_BUCKET}/data/weather_test_3.csv \
                            --model1-path /app/model/model_1.pth \
                            --model2-path /app/model/model_2.pth \
                            --out-dir /app/model
                        """
                    }
                }
            }
        }
        stage('Select Best Model'){
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
                            -e AWS_DEFAULT_REGION=\"${AWS_DEFAULT_REGION}\" \
                            -v \$(pwd)/model:/app/model \
                            ${IMAGE_NAME}:${IMAGE_TAG} \
                            select_best_model.py \
                            --results-dir /app/model/results_test_1.json /app/model/results_test_2.json /app/model/results_test_3.json \
                            --out-dir /app/model/best_model.pth /app/model/results.json
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
