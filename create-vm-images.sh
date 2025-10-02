#!/bin/bash

# ===========================================
# Скрипт создания образа виртуальной машины
# Архивный OCR Сервис
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_NAME="archive-ocr-vm"
VM_DIR="/tmp/vm-build"
BASE_IMAGE="ubuntu-22.04-server-cloudimg-amd64.img"
BASE_IMAGE_URL="https://cloud-images.ubuntu.com/releases/22.04/release/ubuntu-22.04-server-cloudimg-amd64.img"

echo "🏗️ Скрипт создания образа ВМ для Архивного OCR Сервиса"
echo "=================================================="

# Проверка зависимостей
check_dependencies() {
    echo "🔍 Проверка зависимостей..."
    
    command -v qemu-img >/dev/null 2>&1 || { echo "❌ qemu-img не найден. Установите: sudo apt install qemu-utils"; exit 1; }
    command -v virt-customize >/dev/null 2>&1 || { echo "❌ virt-customize не найден. Установите: sudo apt install libguestfs-tools"; exit 1; }
    command -v genisoimage >/dev/null 2>&1 || { echo "❌ genisoimage не найден. Установите: sudo apt install genisoimage"; exit 1; }
    
    echo "✅ Все зависимости найдены"
}

# Создание рабочей директории
create_work_dir() {
    echo "📁 Создание рабочей директории..."
    rm -rf $VM_DIR
    mkdir -p $VM_DIR
    cd $VM_DIR
}

# Скачивание базового образа
download_base_image() {
    echo "⬇️ Скачивание базового образа Ubuntu..."
    if [ ! -f "$BASE_IMAGE" ]; then
        wget -O "$BASE_IMAGE" "$BASE_IMAGE_URL"
    else
        echo "✅ Базовый образ уже существует"
    fi
}

# Создание пользовательских данных cloud-init
create_cloud_init() {
    echo "☁️ Создание конфигурации cloud-init..."
    
    cat > user-data << 'EOF'
#cloud-config
users:
  - default
  - name: archive-ocr
    groups: [sudo, docker]
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys: []

package_update: true
package_upgrade: true

packages:
  - python3
  - python3-pip
  - python3-venv
  - git
  - curl
  - wget
  - htop
  - nginx
  - docker.io
  - docker-compose
  - libgl1-mesa-glx
  - libglib2.0-0
  - libsm6
  - libxext6
  - libxrender-dev
  - libgomp1
  - libgtk-3-0
  - poppler-utils

write_files:
  - path: /opt/install-archive-ocr.sh
    permissions: '0755'
    content: |
      #!/bin/bash
      set -e
      
      echo "🏛️ Установка Архивного OCR Сервиса..."
      
      # Создание директории
      mkdir -p /opt/archive-ocr-service
      cd /opt/archive-ocr-service
      
      # Создание пользователя
      useradd -m -s /bin/bash -G docker archive-ocr || true
      chown archive-ocr:archive-ocr /opt/archive-ocr-service
      
      # Скачивание исходного кода (заглушка)
      # В продакшене замените на реальный репозиторий
      sudo -u archive-ocr mkdir -p uploads results static logs
      
      # Создание виртуального окружения
      sudo -u archive-ocr python3 -m venv venv
      sudo -u archive-ocr ./venv/bin/pip install --upgrade pip
      
      # Создание requirements.txt
      cat > requirements.txt << 'EOL'
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
aiofiles>=23.2.1
sqlalchemy>=2.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
pdf2image>=3.1.0
numpy>=1.24.0
paddleocr>=2.7.0
paddlepaddle>=2.5.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
EOL
      
      # Установка зависимостей
      sudo -u archive-ocr ./venv/bin/pip install -r requirements.txt
      
      # Создание .env файла
      sudo -u archive-ocr cat > .env << 'EOL'
DATABASE_URL=sqlite:///./archive_service.db
MAX_FILE_SIZE=104857600
HOST=0.0.0.0
PORT=8000
LOW_CONFIDENCE_THRESHOLD=0.75
USE_POSTPROCESSING=true
OCR_LANGUAGE=ru
USE_GPU=false
DEBUG=false
LOG_LEVEL=INFO
EOL
      
      # Создание systemd сервиса
      cat > /etc/systemd/system/archive-ocr.service << 'EOL'
[Unit]
Description=Archive OCR Service
After=network.target

[Service]
Type=simple
User=archive-ocr
Group=archive-ocr
WorkingDirectory=/opt/archive-ocr-service
Environment=PATH=/opt/archive-ocr-service/venv/bin
ExecStart=/opt/archive-ocr-service/venv/bin/python web_service.py
Restart=always
RestartSec=3
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOL
      
      # Включение сервиса
      systemctl daemon-reload
      systemctl enable archive-ocr
      
      echo "✅ Архивный OCR Сервис установлен"
      echo "📝 Для загрузки исходного кода выполните:"
      echo "   cd /opt/archive-ocr-service"
      echo "   git clone <your-repo-url> ."
      echo "   systemctl start archive-ocr"

  - path: /etc/nginx/sites-available/archive-ocr
    content: |
      server {
          listen 80;
          server_name _;
          client_max_body_size 100M;
          
          location / {
              proxy_pass http://127.0.0.1:8000;
              proxy_set_header Host $host;
              proxy_set_header X-Real-IP $remote_addr;
              proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
              proxy_set_header X-Forwarded-Proto $scheme;
          }
          
          location /health {
              proxy_pass http://127.0.0.1:8000/health;
          }
      }

runcmd:
  - systemctl enable docker
  - systemctl start docker
  - usermod -aG docker ubuntu
  - usermod -aG docker archive-ocr
  - /opt/install-archive-ocr.sh
  - ln -sf /etc/nginx/sites-available/archive-ocr /etc/nginx/sites-enabled/
  - rm -f /etc/nginx/sites-enabled/default
  - systemctl restart nginx
  - systemctl enable nginx

final_message: |
  🎉 Архивный OCR Сервис готов к работе!
  
  📍 Веб-интерфейс: http://<IP-адрес>:8000
  📍 С Nginx: http://<IP-адрес>
  📍 API документация: http://<IP-адрес>:8000/docs
  
  🔧 Для завершения настройки:
  1. Загрузите исходный код в /opt/archive-ocr-service
  2. Настройте переменные окружения в .env
  3. Запустите сервис: systemctl start archive-ocr
  
  👤 Пользователи:
  - ubuntu (sudo доступ)  
  - archive-ocr (для сервиса)
EOF

    # Создание meta-data
    cat > meta-data << EOF
instance-id: archive-ocr-$(date +%s)
local-hostname: archive-ocr-vm
EOF

    echo "✅ Cloud-init конфигурация создана"
}

# Создание ISO с cloud-init данными
create_cloud_init_iso() {
    echo "💿 Создание ISO с cloud-init данными..."
    genisoimage -output cloud-init.iso -volid cidata -joliet -rock user-data meta-data
}

# Расширение базового образа
resize_base_image() {
    echo "📏 Расширение базового образа до 20GB..."
    cp "$BASE_IMAGE" "${VM_NAME}.qcow2"
    qemu-img resize "${VM_NAME}.qcow2" 20G
}

# Кастомизация образа
customize_image() {
    echo "⚙️ Кастомизация образа..."
    
    virt-customize \
        -a "${VM_NAME}.qcow2" \
        --run-command 'apt-get update && apt-get install -y cloud-init' \
        --run-command 'systemctl enable cloud-init' \
        --run-command 'systemctl enable cloud-config' \
        --run-command 'systemctl enable cloud-final' \
        --run-command 'cloud-init clean' \
        --run-command 'apt-get autoremove -y && apt-get clean' \
        --run-command 'rm -rf /var/lib/apt/lists/*' \
        --run-command 'history -c && history -w'
}

# Создание дополнительных форматов
create_additional_formats() {
    echo "🔄 Создание дополнительных форматов образов..."
    
    # VMDK для VMware
    echo "📦 Создание VMDK образа..."
    qemu-img convert -f qcow2 -O vmdk -o subformat=streamOptimized \
        "${VM_NAME}.qcow2" "${VM_NAME}.vmdk"
    
    # RAW образ
    echo "📦 Создание RAW образа..."
    qemu-img convert -f qcow2 -O raw "${VM_NAME}.qcow2" "${VM_NAME}.raw"
    
    # Сжатие RAW образа
    echo "🗜️ Сжатие RAW образа..."
    gzip -9 "${VM_NAME}.raw"
    
    # VDI для VirtualBox
    echo "📦 Создание VDI образа..."
    qemu-img convert -f qcow2 -O vdi "${VM_NAME}.qcow2" "${VM_NAME}.vdi"
}

# Создание контрольных сумм
create_checksums() {
    echo "🔐 Создание контрольных сумм..."
    
    for file in "${VM_NAME}".{qcow2,vmdk,raw.gz,vdi}; do
        if [ -f "$file" ]; then
            sha256sum "$file" > "$file.sha256"
            md5sum "$file" > "$file.md5"
        fi
    done
}

# Создание информационного файла
create_info_file() {
    echo "📄 Создание информационного файла..."
    
    cat > "${VM_NAME}-info.txt" << EOF
🏛️ АРХИВНЫЙ OCR СЕРВИС - ОБРАЗЫ ВИРТУАЛЬНЫХ МАШИН
==================================================

📅 Дата создания: $(date)
🖥️ Базовая ОС: Ubuntu 22.04 LTS Server
💾 Размер диска: 20GB
🧠 Рекомендуемая RAM: 8GB+
⚙️ Рекомендуемые CPU: 4+ ядра

📦 ДОСТУПНЫЕ ФОРМАТЫ:
- ${VM_NAME}.qcow2     - QEMU/KVM формат (рекомендуется)
- ${VM_NAME}.vmdk      - VMware формат
- ${VM_NAME}.raw.gz    - RAW формат (сжатый)
- ${VM_NAME}.vdi       - VirtualBox формат

🔐 КОНТРОЛЬНЫЕ СУММЫ:
Файлы *.sha256 и *.md5 содержат контрольные суммы для проверки целостности.

👤 УЧЕТНЫЕ ЗАПИСИ:
- ubuntu       (пароль: не задан, только SSH ключи)
- archive-ocr  (системный пользователь для сервиса)

🌐 СЕТЕВАЯ КОНФИГУРАЦИЯ:
- IP: получается по DHCP
- Порты: 22 (SSH), 80 (HTTP), 8000 (API)

🚀 БЫСТРЫЙ ЗАПУСК:

1. QEMU/KVM:
   qemu-system-x86_64 -m 8G -smp 4 -hda ${VM_NAME}.qcow2 -netdev user,id=net0 -device virtio-net-pci,netdev=net0

2. VMware:
   - Импортируйте ${VM_NAME}.vmdk как новую ВМ
   - Выделите 8GB RAM и 4 CPU

3. VirtualBox:
   - Создайте новую ВМ с ${VM_NAME}.vdi
   - Настройте 8GB RAM и 4 CPU

📝 ЗАВЕРШЕНИЕ НАСТРОЙКИ:

После первого запуска:
1. Подключитесь по SSH как ubuntu
2. Перейдите в /opt/archive-ocr-service
3. Загрузите исходный код проекта
4. Настройте .env файл
5. Запустите: sudo systemctl start archive-ocr

🔗 ДОСТУП К СЕРВИСУ:
- Веб-интерфейс: http://<IP-адрес>
- API: http://<IP-адрес>:8000
- Документация: http://<IP-адрес>:8000/docs

⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:
- Первый запуск может занять 5-10 минут (загрузка OCR моделей)
- Рекомендуется SSD диск для лучшей производительности
- GPU ускорение поддерживается при наличии CUDA

🆘 ПОДДЕРЖКА:
Документация: deployment-guide.pdf
Техническая поддержка: ЛЦТ 2025

EOF
}

# Тестирование образа
test_image() {
    echo "🧪 Тестирование образа..."
    
    echo "📊 Информация об образе QCOW2:"
    qemu-img info "${VM_NAME}.qcow2"
    
    echo "📁 Размеры файлов:"
    ls -lh "${VM_NAME}".*
    
    echo "✅ Тестирование завершено"
}

# Упаковка результатов
package_results() {
    echo "📦 Упаковка результатов..."
    
    mkdir -p ../release
    
    # Создание архива с образами
    tar -czf "../release/archive-ocr-vm-images-$(date +%Y%m%d).tar.gz" \
        "${VM_NAME}".* \
        cloud-init.iso \
        user-data \
        meta-data
    
    # Копирование файлов в release директорию
    cp "${VM_NAME}".* "../release/"
    cp "${VM_NAME}-info.txt" "../release/README.txt"
    
    echo "✅ Результаты сохранены в ../release/"
}

# Главная функция
main() {
    echo "🎯 Начинаем создание образов ВМ..."
    
    check_dependencies
    create_work_dir
    download_base_image
    create_cloud_init
    create_cloud_init_iso
    resize_base_image
    customize_image
    create_additional_formats
    create_checksums
    create_info_file
    test_image
    package_results
    
    echo ""
    echo "🎉 СОЗДАНИЕ ОБРАЗОВ ЗАВЕРШЕНО!"
    echo "=================================================="
    echo "📁 Результаты находятся в: $(pwd)/../release/"
    echo "📋 Инструкции: README.txt"
    echo "🔐 Проверьте контрольные суммы перед развертыванием"
    echo ""
    echo "🚀 Готовые образы:"
    echo "   - QEMU/KVM: ${VM_NAME}.qcow2"
    echo "   - VMware: ${VM_NAME}.vmdk"  
    echo "   - VirtualBox: ${VM_NAME}.vdi"
    echo "   - RAW: ${VM_NAME}.raw.gz"
    echo ""
    echo "📖 Подробные инструкции в deployment-guide.pdf"
}

# Запуск
main "$@"