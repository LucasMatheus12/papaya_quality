from django.db import models


class TrainingTool(models.Model):
    date_update = models.DateTimeField(
        verbose_name='Data de atualização',
        auto_now=True
    )
    
    date_creation = models.DateTimeField(
        verbose_name='Data de criação',
        auto_now_add=True
    )
    
    model = models.FileField(
        upload_to='models_train/',
        verbose_name='Modelo'
    )

    graph_acuracia = models.ImageField(
        upload_to='images_accuracy/',
        verbose_name='Gráfico de acurácia',
        null=True, blank=True
    )

    graph_confusion_matrix = models.ImageField(
        upload_to='graphs_cm/',
        verbose_name='Gráfico de matriz de confusão',
        null=True, blank=True
    )

    historico = models.ImageField(
        upload_to='historico/',
        verbose_name='Histórico',
        null=True, blank=True
    )


    def __str__(self):
        '''Método que retorna a representação do objeto como string.'''
        return f"{self.id}"

    class Meta:
        '''Sub classe para definir meta atributos da classe principal.'''

        app_label = 'training_tool'
        verbose_name = 'Training Tool'
        verbose_name_plural = 'Training Tools'
