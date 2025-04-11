output "db_endpoint" {
  description = "Endpoint of the database"
  value       = aws_db_instance.main.endpoint
}

output "db_name" {
  description = "Name of the database"
  value       = aws_db_instance.main.db_name
}

output "db_username" {
  description = "Username of the database"
  value       = aws_db_instance.main.username
  sensitive   = true
}
