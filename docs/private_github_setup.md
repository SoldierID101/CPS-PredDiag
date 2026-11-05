# CPS-PredDiag — безопасная первая публикация в PRIVATE GitHub

**Дата:** 05.11.2026. **Локальный каталог:** `D:\CPS-PredDiag`.

## До первого push

1. Сохраните локальную резервную копию проекта. Git — история версий, а не замена независимому бэкапу данных.
2. Распакуйте `.gitignore` и `docs/` в корень `D:\CPS-PredDiag`.
   Если `.gitignore` уже существует, **объедините правила**, не уничтожая прежние.
3. Не загружайте `dataset/`, `data/`, Parquet, CSV результатов, `.venv/`,
   `outputs/`, пароли, `.env`, токены, дампы БД и сериализованные модели.
4. `.gitignore` не удаляет из истории файлы, уже попавшие в коммит.
   Если обнаружен секрет в коммите — **остановитесь до push**, перевыпустите секрет,
   удалите его из истории. Одного удаления файла недостаточно.
5. Локально в PowerShell:

```powershell
cd D:\CPS-PredDiag
git --version
git rev-parse --is-inside-work-tree
git status --short
git remote -v
```

Если `git rev-parse` сообщает «not a git repository», это нормально для нового проекта.
Если уже есть Git — **не вызывайте `git init` повторно**, сначала проверьте существующие remotes
и историю, чтобы случайно не отправить код в старый или публичный репозиторий.

## Создание приватного репозитория

1. Перейдите на https://github.com/new .
2. Укажите имя `CPS-PredDiag`, владельца — личную учётную запись.
3. Отметьте **Private** (не Public).
4. Не включайте шаблонный README, Gitignore или лицензию: чтобы новый remote был пуст.
5. Создайте репозиторий. Убедитесь, что у него отметка **Private**.
6. Никому не выдавайте доступ через Settings → Collaborators, пока это не требуется.

## Если локального `.git` ещё нет

```powershell
cd D:\CPS-PredDiag
git init -b main
```

Проверьте местный `.gitignore`, затем добавьте только проверенные текстовые файлы.
Ниже безопасный ориентир; **при необходимости добавьте отдельно свои JSON-конфигурации**.

```powershell
git add -- .gitignore
git add -- '*.py' '*.md' '*.puml' '*.toml' '*.yaml' '*.yml'
git add -- 'requirements.txt' 'pyproject.toml'
git status --short
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
```

`git add` может сообщить, что отдельных необязательных файлов не найдено —
это не проблема; добавьте только те, что реально существуют. Для конфигураций:
например, `git add rules/class2_dhsv.json`, после проверки содержимого.

**Обязательно проверьте весь список `git diff --cached --name-only`**.
Если случайно добавили файл, уберите из индекса (локальный файл сохранится):

```powershell
git restore --staged -- path/to/unwanted_file
```

Для уже закоммиченного файла `git restore --staged` не очищает историю.
Перед первым push секретный коммит потребует отдельного исправления.

Внимательно прочитайте и содержимое:

```powershell
git diff --cached
```

Только после этого:

```powershell
git commit -m "Record CPS-PredDiag research baseline and experiments"
git branch -M main
```

## Связывание с PRIVATE GitHub

Скопируйте **точный HTTPS URL только что созданного PRIVATE репозитория**.
Не используйте URL старого ForecastingAgent.

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/CPS-PredDiag.git
git remote -v
git push -u origin main
```

В примере замените `YOUR_ACCOUNT` на ваш GitHub username.
**Не отправляйте `git push`, пока не проверили, что `origin` указывает именно
на новый PRIVATE репозиторий.**

GitHub может открыть окно входа в браузере. Не помещайте личный токен в текст
команды и не отправляйте токены в чат.

## Контроль после отправки

```powershell
git status
git log -1 --oneline
git remote -v
git ls-files
```

В браузере откройте страницу репозитория и убедитесь:
- видна отметка **Private**;
- есть код Python, каталог `tests`, документация `docs`;
- нет каталогов `dataset`, `outputs`, `.venv`, `.env`, моделей, больших CSV;
- нет нежелательных участников с доступом.

**Ограничение:** приватный репозиторий доступен вам, GitHub как сервису и
уполномоченным соавторам при наличии разрешения. Это не локальное зашифрованное
хранилище и не гарантирует сокрытие уже опубликованных ранее копий.

## Если в проекте уже есть Git или `origin`

Не запускайте команды изменения remote вслепую. Получите:

```powershell
git status -sb
git remote -v
git log -3 --oneline
```

Если `origin` указывает на старый публичный проект, сначала выясните,
как сохранить нужную историю. Безопаснее не менять видимость старого репозитория,
а создать **новый PRIVATE** и подключить его только после проверки содержимого
и истории будущего push.

## Полезные официальные инструкции

- https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility
- https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
